from pathlib import Path
import json
import numpy as np

base = Path('E:/codex-artifacts')
old = base / 'sonic23_teleop_six_hour_20260910'
new = base / 'sonic23_teleop_resume_20260911'
paths = [old / 'bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1/trace.npz',
         old / 'mjbatch_full_v1/pico_v4_native323_freshseed_allmargin_5iter_full_v1/trace.npz',
         new / 'hard_feasibility_restoration_continuation_3740_v1/trace.npz',
         new / 'recovery_probe_v1/actual_3740_bfm_history.npz',
         new / 'recovery_probe_v1/actual_3740_integration_state.npz']
for p in paths:
    with np.load(p, allow_pickle=False) as a:
        print(json.dumps(dict(path=str(p), arrays={k:list(a[k].shape) for k in a.files}), indent=2))
