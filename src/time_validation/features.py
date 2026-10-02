"""Reuse original feature definitions, with origin-specific data and global encodings."""
import gc
import importlib.util
import json
import time
import numpy as np
import pandas as pd
from common import ROOT, META, FILES, stage_root, artifact_id, complete, checkpoint, event, atomic_json

def legacy(cutoff):
    spec=importlib.util.spec_from_file_location('base_features',ROOT/'src/1_preprocessing_by_store.py')
    p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
    p.END_TRAIN=cutoff; p.TOTAL_DAYS=cutoff+28; p.ENCODING_END=cutoff
    return p

def inputs(c, stage):
    cut=c['stages'][stage]; p=legacy(cut)
    cal=pd.read_csv(ROOT/'data/calendar.csv')
    cal['d_num']=cal.d.str.removeprefix('d_').astype(np.int16)
    cal=cal.loc[cal.d_num<=cut+28].copy()
    meta=pd.read_csv(ROOT/'data/sales_train_evaluation.csv', usecols=META, dtype='string')
    voc=p.make_vocabs(meta,cal)
    dtype={k:pd.CategoricalDtype(voc[k]) for k in META}
    days=[f'd_{d}' for d in range(1,cut+1)]
    dtype.update({d:np.float32 for d in days})
    sales=pd.read_csv(ROOT/'data/sales_train_evaluation.csv', usecols=META+days, dtype=dtype)
    prices=pd.read_csv(ROOT/'data/sell_prices.csv', dtype={'store_id':dtype['store_id'],'item_id':dtype['item_id'],'wm_yr_wk':np.int16,'sell_price':np.float32})
    prices=prices.loc[prices.wm_yr_wk.isin(cal.wm_yr_wk.unique())].sort_values(['store_id','item_id','wm_yr_wk']).reset_index(drop=True)
    assert not prices.duplicated(['store_id','item_id','wm_yr_wk']).any()
    release=prices.groupby(['store_id','item_id'],observed=True).wm_yr_wk.min().rename('release').reset_index()
    for k in ['event_name_1','event_type_1','event_name_2','event_type_2','snap_CA','snap_TX','snap_WI']:
        cal[k]=p.as_global_category(cal[k],voc[k])
    ordinals={id:i for i,id in enumerate(meta.id)}
    return p,sales,prices,cal,voc,release,ordinals

def encoding_stats(p,sales,cal,release,cut):
    # Sufficient statistics over all stores, without a 59M-row long table.
    # Match the original release filtering; use only d<=this origin.
    rows=sales[META].merge(release,on=['store_id','item_id'],how='left',validate='one_to_one',sort=False)
    assert rows.id.astype(str).tolist()==sales.id.astype(str).tolist()
    y=sales[[f'd_{d}' for d in range(1,cut+1)]].to_numpy(dtype=np.float64)
    weeks=cal.set_index('d_num').loc[np.arange(1,cut+1),'wm_yr_wk'].to_numpy()
    mask=weeks[None,:]>=rows.release.to_numpy()[:,None]
    rows['count']=mask.sum(axis=1)
    y[~mask]=0
    rows['total']=y.sum(axis=1); rows['sumsq']=np.einsum('ij,ij->i',y,y)
    del y,mask
    stats={}
    for keys in p.ENCODING_GROUPS:
        s=rows.groupby(keys,observed=True)[['count','total','sumsq']].sum().reset_index()
        s['mean']=s.total/s['count'].replace(0,np.nan)
        var=(s.sumsq-s.total**2/s['count'].replace(0,np.nan)).clip(lower=0)/(s['count']-1).replace(0,np.nan)
        s['std']=np.sqrt(var).where(s['count']>=2)
        stats[tuple(keys)]=s[[*keys,'count','mean','std']]
    return stats

def prepare(c,stage,store):
    dest=stage_root(c,stage)/'cache'/store; dest.mkdir(parents=True,exist_ok=True)
    fp=artifact_id(c,stage,store,'cache')
    if complete(dest/'complete.json',fp,FILES):
        event('cache_verified',stage=stage,store=store); return
    started=time.monotonic(); cut=c['stages'][stage]
    p,sales,prices,cal,voc,release,ordinals=inputs(c,stage)
    stats=encoding_stats(p,sales,cal,release,cut)
    if c.get('smoke_items'):
        chosen=sales.loc[sales.store_id==store,'id'].iloc[:c['smoke_items']]
        sales=sales.loc[sales.id.isin(chosen)].copy()
    grid=p.build_base_grid(store,sales,cal,voc,ordinals,int(release.release.min()),release)
    del sales; gc.collect()
    assert grid.index.is_unique and grid.index.is_monotonic_increasing
    assert grid.loc[grid.d>cut,'sales'].isna().all()
    assert grid.loc[grid.d<=cut,'sales'].notna().all()
    future=grid.loc[grid.d>cut]
    assert future.groupby('id',observed=True).size().eq(28).all()
    expected=c.get('smoke_items',3049)
    assert future.id.nunique()==expected
    p.atomic_pickle(grid,dest/FILES[0])
    for name,build in [
        (FILES[1],lambda:p.build_price_table(grid,p.make_price_features(store,prices,cal),cal)),
        (FILES[2],lambda:p.build_calendar_table(grid,cal,voc)),
        (FILES[3],lambda:p.build_lag_table(grid)),
        (FILES[4],lambda:p.build_encoding_table(grid,stats))]:
        frame=build()
        assert frame.index.equals(grid.index)
        assert frame[['id','d']].equals(grid[['id','d']])
        p.atomic_pickle(frame,dest/name)
        del frame; gc.collect()
    atomic_json(voc,dest/'categories.json')
    checkpoint(dest/'complete.json',fp,FILES,cutoff=cut,first_day=1,last_day=cut+28,
               rows=len(grid),series=expected,encoding_end=cut,encoding_stores=10,
               seconds=time.monotonic()-started,price_week_max=int(prices.wm_yr_wk.max()))
    event('features_complete',stage=stage,store=store,seconds=time.monotonic()-started)
