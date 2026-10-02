"""Read-only external reference audit; downloads source, executes only inspected aggregator AST.

No third-party package is installed and no raw data is modified.
"""
import ast
import hashlib
import json
import urllib.request
import numpy as np
import pandas as pd
from common import ROOT, atomic_json
from metrics import WRMSSE
from tests import metadata

url='https://raw.githubusercontent.com/Nixtla/datasetsforecast/main/datasetsforecast/m5.py'
source=urllib.request.urlopen(url,timeout=30).read()
tree=ast.parse(source)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='M5Evaluation')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='aggregate_levels')
# Only the already-reviewed pure pandas grouping routine is used, without imports or I/O.
allowed={'copy','assign','groupby','sum','reset_index','items','zip','dict','rename','insert','append','concat','fillna','set_index','range'}
for node in ast.walk(method):
    if isinstance(node,ast.Call):
        name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
        assert name in allowed,f'External reference changed: unexpected call {name}'
method.decorator_list=[]
level_assignment=next(n for n in cls.body if isinstance(n,ast.AnnAssign) and n.target.id=='levels')
levels=eval(compile(ast.Expression(level_assignment.value),'<reference levels>','eval'),{'dict':dict})
namespace={'np':np,'pd':pd,'M5Evaluation':type('M5Evaluation',(),{'levels':levels})}
exec(compile(ast.Module(body=[method],type_ignores=[]),'<reviewed reference aggregator>','exec'),namespace)
aggregate=namespace['aggregate_levels']
meta=metadata(); rng=np.random.default_rng(829)
history=rng.poisson(2,(8,90)).astype(float); history[:,:4]=0
truth=rng.poisson(2,(8,28)).astype(float); pred=rng.uniform(0,4,(8,28)); revenue=np.arange(1,9,dtype=float)
def frame(values):
    return pd.concat([meta.rename(columns={'id':'unique_id'}),pd.DataFrame(values,columns=[f'd_{d+1}' for d in range(values.shape[1])])],axis=1)
train=aggregate(frame(history)); target=aggregate(frame(truth)); forecast=aggregate(frame(pred))
dollars=aggregate(frame(revenue[:,None])).iloc[:,0]
def scalar_scale(row):
    x=row.to_numpy(); x=x[np.argmax(x!=0):]
    return ((x[1:]-x[:-1])**2).mean()
scale=train.apply(scalar_scale,axis=1)
weights=dollars/dollars.groupby(level='Level_id').transform('sum')
reference=((((target-forecast)**2).mean(axis=1)/scale)**0.5*weights).groupby(level='Level_id').sum()
score,levels_actual=WRMSSE(meta,history,revenue).score(truth,pred)
expected=reference.reindex([f'Level{i}' for i in range(1,13)]).to_numpy()
np.testing.assert_allclose(levels_actual.wrmsse,expected,rtol=1e-12,atol=1e-12)
np.testing.assert_allclose(score,reference.mean(),rtol=1e-12)
record={'reference_url':url,'source_sha256':hashlib.sha256(source).hexdigest(),
        'reference_score':float(reference.mean()),'implementation_score':score,
        'maximum_level_difference':float(np.max(np.abs(expected-levels_actual.wrmsse))),
        'method':'external aggregate_levels AST + reference scalar scale formula, 12 levels',
        'pass':True}
atomic_json(record,ROOT/'docs/wrmsse_reference_check.json'); print(json.dumps(record,indent=2))
