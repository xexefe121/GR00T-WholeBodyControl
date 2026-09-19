"""Persistent eight-worker preview; private per-world MuJoCo scratch states."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from gear_sonic.utils.g1_true23_native_preview_guard import NativePreviewGuard


class BatchNativePreview:
    def __init__(self,model,contract,count,library,delay=6,workers=8):
        if not 1<=workers<=8:raise ValueError('Preview requires 1..8 outer workers')
        self.guards=[NativePreviewGuard(model,contract,delay_substeps=delay,library=library,native_parallelism=1)
                     for _ in range(count)]
        self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='native23-preview')
        self.last_status=[]

    def apply(self,qpos,qvel,requested,previous,valid):
        def evaluate(i):
            return self.guards[i].apply(qpos[i],qvel[i],requested[i],previous[i] if valid[i] else None)
        results=list(self.pool.map(evaluate,range(len(self.guards))))
        self.last_status=[status for _,status in results]
        return np.stack([target for target,_ in results])

    def close(self):self.pool.shutdown(wait=True)
