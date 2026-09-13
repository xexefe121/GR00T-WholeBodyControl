import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
with np.load(a.output/'trace.npz') as z:
    epoch=int(z['epoch_ns']);timing=z['timing']
for name in ('standing','active'):
    rows=np.load(a.output/(name+'_calls.npy'))
    if not len(rows):continue
    elapsed=(rows[:,3]-rows[:,2])/1e6
    finish=(rows[:,3]-(epoch+(rows[:,0]+1)*20_000_000))/1e6
    missed=rows[finish>0]
    print(json.dumps(dict(worker=name,calls=len(rows),inference_ms=np.percentile(elapsed,[50,95,99,100]).tolist(),
        deadline_misses=int((finish>0).sum()),missing_controls=np.setdiff1d(np.arange(3069),rows[:,0]).tolist()[:60],
        missed_first_rows=[dict(control=int(r[0]),work_ms=(r[3]-r[2])/1e6,queue_ms=(r[2]-r[1])/1e6,
            finish_late_ms=(r[3]-(epoch+(r[0]+1)*20_000_000))/1e6,published=int(r[4])) for r in missed[:12]])))
