import gc
import json
import os
import resource
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from common import F, FILES, stage_root, manifest, complete, checkpoint, atomic_csv, atomic_json, event

MEAN={'recursive':['enc_cat_id_mean','enc_cat_id_std','enc_dept_id_mean','enc_dept_id_std','enc_item_id_mean','enc_item_id_std'],
      'nonrecursive':['enc_store_id_dept_id_mean','enc_store_id_dept_id_std','enc_item_id_state_id_mean','enc_item_id_state_id_std']}
REMOVE={'id','state_id','store_id','date','wm_yr_wk','d','sales'}
ROLL=[(s,w) for s in [1,7,14] for w in [7,14,30,60]]

def load_frame(c,stage,store,mode):
    dest=stage_root(c,stage)/'cache'/store
    assert complete(dest/'complete.json',manifest(c),FILES)
    base=pd.read_pickle(dest/FILES[0]); reference=base.index
    for name in FILES[1:]:
        table=pd.read_pickle(dest/name)
        assert table.index.equals(reference)
        assert table[['id','d']].equals(base[['id','d']])
        cols=[col for col in table if col not in {'id','d','sales'}]
        if name==FILES[4]: cols=MEAN[mode]
        if mode=='nonrecursive': cols=[col for col in cols if '_tmp_' not in col]
        for col in cols:
            assert col not in base
            base[col]=table[col]
        del table; gc.collect()
    features=[col for col in base if col not in REMOVE]
    # Construct a contiguous frame once, rather than repeated full-data copies.
    return base,features

def predict_recursive(model,frame,features,cut,threads):
    history=frame.loc[frame.d>cut-100,['id','d','sales']].copy()
    history['sales']=history.sales.astype(np.float32)
    history.loc[history.d>cut,'sales']=np.float32(np.nan)
    assert history.loc[history.d>cut,'sales'].isna().all()
    future=frame.loc[frame.d>cut,['id','d',*features]].copy()
    # Pure positional history matrix keyed by explicit id and day. Each day recomputes
    # short rolling features from observed history plus earlier predictions only.
    ids=pd.Index(future.id.astype(str).drop_duplicates())
    days=np.arange(cut-99,cut+29)
    wide=history.assign(id=history.id.astype(str)).pivot(index='id',columns='d',values='sales').reindex(index=ids,columns=days)
    values=wide.to_numpy(dtype=np.float32)
    result=np.empty((len(ids),28),dtype=np.float64)
    for h in range(28):
        day=cut+h+1; position=100+h
        rows=future.loc[future.d==day].copy()
        order=ids.get_indexer(rows.id.astype(str)); assert (order>=0).all()
        for shift,window in ROLL:
            start=position-shift-window+1; end=position-shift+1
            # np.mean deliberately requires the entire window, matching pandas min_periods.
            rows[f'rolling_mean_tmp_{shift}_{window}']=values[order,start:end].mean(axis=1,dtype=np.float64).astype(np.float32)
        pred=np.asarray(model.predict(rows[features],num_threads=threads),dtype=np.float64)
        assert np.isfinite(pred).all() and (pred>=0).all()
        values[order,position]=pred.astype(np.float32)
        result[order,h]=pred
    return pd.DataFrame(result,index=ids,columns=F).rename_axis('id').reset_index()

def run_model(c,stage,store,mode):
    dest=stage_root(c,stage)/mode/store; dest.mkdir(parents=True,exist_ok=True)
    fp=manifest(c); products=['model.txt','predictions.csv','importance.csv','metadata.json']
    if complete(dest/'complete.json',fp,products):
        event('model_verified',stage=stage,store=store,mode=mode); return
    started=time.monotonic(); cut=c['stages'][stage]
    frame,features=load_frame(c,stage,store,mode)
    train_mask=frame.d.le(cut); forecast_mask=frame.d.gt(cut)&frame.d.le(cut+28)
    assert not (train_mask & forecast_mask).any()
    assert frame.loc[train_mask,'sales'].notna().all()
    assert frame.loc[forecast_mask,'sales'].isna().all()
    assert frame.loc[train_mask,'d'].max()==cut
    assert frame.loc[forecast_mask,'d'].min()==cut+1
    # No overlapping LightGBM valid_set. Holdout is scored only after 28-day inference.
    params={**c['params'],**c['mode_params'][mode],'num_threads':c['threads']}
    train_rows=int(train_mask.sum())
    future=frame.loc[frame.d>cut-100].copy()
    train=frame.loc[train_mask,features]
    labels=frame.loc[train_mask,'sales'].to_numpy(dtype=np.float32)
    del frame; gc.collect()
    dataset=lgb.Dataset(train,label=labels,params=params,free_raw_data=True)
    dataset.construct()
    del train,labels; gc.collect()
    train_start=time.monotonic()
    event('training_started',stage=stage,store=store,mode=mode,rows=train_rows,features=len(features),rounds=c['rounds'])
    def progress(env):
        if (env.iteration+1)%100==0:
            event('iteration',stage=stage,store=store,mode=mode,iteration=env.iteration+1,seconds=time.monotonic()-train_start)
    estimator=lgb.train(params,dataset,num_boost_round=c['rounds'],callbacks=[progress])
    train_seconds=time.monotonic()-train_start
    estimator.save_model(str(dest/'model.txt.tmp')); os.replace(dest/'model.txt.tmp',dest/'model.txt')
    del dataset; gc.collect()
    predict_start=time.monotonic()
    if mode=='recursive': output=predict_recursive(estimator,future,features,cut,c['threads'])
    else:
        rows=future.loc[future.d>cut].copy()
        rows['prediction']=np.asarray(estimator.predict(rows[features],num_threads=c['threads']),dtype=np.float64)
        wide=rows.pivot(index='id',columns='d',values='prediction')
        assert wide.columns.tolist()==list(range(cut+1,cut+29))
        wide.columns=F; output=wide.reset_index()
    assert output.id.nunique()==c.get('smoke_items',3049)
    assert np.isfinite(output[F].to_numpy()).all() and (output[F].to_numpy()>=0).all()
    atomic_csv(output,dest/'predictions.csv')
    atomic_csv(pd.DataFrame({'feature':features,'gain':estimator.feature_importance('gain'),'split':estimator.feature_importance('split')}).sort_values('gain',ascending=False),dest/'importance.csv')
    atomic_json(dict(features=features,params=params,rounds=c['rounds'],train_rows=train_rows,
                     train_first_day=1,train_last_day=cut,predict_days=[cut+1,cut+28],
                     train_seconds=train_seconds,predict_seconds=time.monotonic()-predict_start,
                     total_seconds=time.monotonic()-started,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),dest/'metadata.json')
    checkpoint(dest/'complete.json',fp,products)
    event('model_complete',stage=stage,store=store,mode=mode,seconds=time.monotonic()-started,train_seconds=train_seconds)
