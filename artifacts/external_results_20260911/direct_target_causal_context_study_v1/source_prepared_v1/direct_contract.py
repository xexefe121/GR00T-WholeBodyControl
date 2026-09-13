"""Pure direct-target feature, label, output and weighting contract."""
import numpy as np

RETAINED=np.r_[np.arange(52),np.arange(75,1023)].astype(np.int64)
DATASETS=('old','query1','query250','pico','walk002')
COUNTS=((100,819,100),)*3+((100,5780,100),(100,667,100))
PHYSICAL_COUNTS=(99,819,100)*3
SEED=20260911
UPDATES=5000

def exact(a,b,name=''):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():raise AssertionError(name)

def reduce_features(values):
    if values.shape[-1]!=1069 or values.dtype!=np.float32:raise ValueError('Expected original float32 1069 features.')
    return np.take(values,RETAINED,axis=-1)

def nominal_cells(dataset,phase,counts=COUNTS):
    cells=[np.flatnonzero((dataset==d)&(phase==p)) for d in range(len(counts)) for p in range(3)]
    if [len(ids) for ids in cells]!=[n for row in counts for n in row]:raise ValueError('Nominal dataset/phase counts differ.')
    if sum(map(len,cells))!=len(dataset):raise ValueError('Rows missing from nominal cells.')
    return cells

def weighted_normalization(features,cells,floor=.05):
    """Equal cell population moments, including between-cell mean differences."""
    if features.dtype!=np.float32 or not np.isfinite(features).all():raise ValueError('Invalid features.')
    if any(len(ids)==0 for ids in cells):raise ValueError('Empty normalization cell.')
    means=np.stack([np.mean(features[ids],axis=0,dtype=np.float64) for ids in cells])
    mean64=np.mean(means,axis=0,dtype=np.float64)
    variance64=np.mean(np.stack([np.mean((features[ids].astype(np.float64)-mean64)**2,axis=0) for ids in cells]),axis=0)
    std64=np.maximum(np.sqrt(variance64),np.float64(floor))
    return mean64.astype(np.float32),std64.astype(np.float32),mean64,variance64

def normalized_labels(target,default,span32):
    if target.dtype!=np.float64 or default.dtype!=np.float64 or span32.dtype!=np.float32:raise ValueError('Label cast contract.')
    return ((target-default)/span32.astype(np.float64)).astype(np.float32)

def absolute_targets(normalized,default,span32,limits):
    if normalized.dtype!=np.float32 or default.dtype!=np.float64 or span32.dtype!=np.float32 or limits.dtype!=np.float64:raise ValueError('Runtime cast contract.')
    raw=default+span32.astype(np.float64)*normalized.astype(np.float64)
    return raw,np.clip(raw,limits[:,0],limits[:,1])

def cosine_rate(index,updates=UPDATES):
    import math
    if not 0<=index<updates:raise ValueError('LR index outside fixed updates.')
    return 3e-5+.5*(3e-4-3e-5)*(1+math.cos(math.pi*index/(updates-1)))
