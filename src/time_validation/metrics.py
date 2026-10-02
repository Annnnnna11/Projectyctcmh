"""12-level WRMSSE; independent grouping reference lives in tests.py.

Definition references: Mcompetitions/M5-methods and Nixtla/datasetsforecast/m5.py.
Exclude leading zero demand, INCLUDING the transition into the first sale.
Each of twelve levels has dollar weights summing to one; average level scores.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

LEVELS=[[],['state_id'],['store_id'],['cat_id'],['dept_id'],
        ['state_id','cat_id'],['state_id','dept_id'],['store_id','cat_id'],
        ['store_id','dept_id'],['item_id'],['state_id','item_id'],['item_id','store_id']]
COUNTS=[1,3,10,3,7,9,21,30,70,3049,9147,30490]

def scaling(history):
    n,t=history.shape; out=np.zeros(n,dtype=np.float64)
    for start in range(0,n,512):
        y=np.asarray(history[start:start+512],dtype=np.float64)
        first=(y!=0).argmax(axis=1)
        count=t-first-1
        diff=np.diff(y,axis=1)**2
        diff[np.arange(t-1)[None,:]<first[:,None]]=0
        np.divide(diff.sum(axis=1),count,out=out[start:start+len(y)],where=count>0)
    return out

def rmsse(mse,scale):
    out=np.zeros_like(mse,dtype=float)
    np.divide(mse,scale,out=out,where=scale>0)
    out[(scale<=0)&(mse>0)]=np.inf
    return np.sqrt(out)

class WRMSSE:
    def __init__(self,meta,history,revenue,full=False):
        assert meta.id.is_unique and len(meta)==len(history)==len(revenue)
        assert np.isfinite(history).all() and (history>=0).all()
        assert np.isfinite(revenue).all() and (revenue>=0).all()
        if revenue.sum()<=0: raise ValueError('WRMSSE undefined: zero total dollar sales')
        self.levels=[]
        for level,keys in enumerate(LEVELS,1):
            if keys:
                codes,groups=pd.factorize(pd.MultiIndex.from_frame(meta[keys]),sort=True)
            else:
                codes=np.zeros(len(meta),dtype=int); groups=['Total']
            agg=csr_matrix((np.ones(len(meta)),(codes,np.arange(len(meta)))),shape=(len(groups),len(meta)))
            if full: assert len(groups)==COUNTS[level-1],(level,len(groups))
            scale=scaling(agg@history)
            dollars=agg@revenue; weight=dollars/dollars.sum()
            assert np.isclose(weight.sum(),1)
            self.levels.append((agg,scale,weight))
    def score(self,truth,prediction):
        assert truth.shape==prediction.shape
        assert np.isfinite(prediction).all()
        rows=[]
        for level,(agg,scale,weight) in enumerate(self.levels,1):
            error=agg@(prediction-truth)
            score=rmsse(np.mean(error**2,axis=1),scale)
            positive=weight>0
            value=float(np.sum(weight[positive]*score[positive]))
            rows.append(dict(level=level,series=len(scale),wrmsse=value,
                             zero_scale=int((scale==0).sum()),zero_weight=int((weight==0).sum()),
                             positive_weight_zero_scale=int(((scale==0)&positive).sum())))
        return float(np.mean([r['wrmsse'] for r in rows])),pd.DataFrame(rows)

def errors(y,p):
    e=np.asarray(p,dtype=float)-y
    return dict(MAE=float(np.abs(e).mean()),RMSE=float(np.sqrt((e**2).mean())),
                bias=float(e.mean()),bias_total=float(e.sum()),
                bias_percent=float(100*e.sum()/y.sum()) if y.sum() else None)

def aligned(frame,ids):
    from common import F
    assert frame.columns.tolist()==['id',*F]
    assert frame.id.is_unique and set(frame.id)==set(ids)
    values=frame.set_index('id').loc[ids,F].to_numpy(dtype=np.float64)
    assert values.shape==(len(ids),28)
    assert np.isfinite(values).all() and (values>=0).all()
    return values
