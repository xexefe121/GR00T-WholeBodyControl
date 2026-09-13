"""Pure artifact path and serialization helpers."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
BASE=Path(__file__).resolve().parent.parent
if sys.platform=="win32":
    ROOT=Path("Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof")
    TASK=Path("E:/codex-artifacts/sonic23_teleop_six_hour_20260910")
    DEPS=None
else:
    ROOT=Path("/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof")
    TASK=Path("/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910")
    DEPS=Path("/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps")
BUNDLE=ROOT/"artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1"
ONNX=ROOT/"artifacts/teleop_six_hour_20260910/bfm_onnx_v2"
REFERENCE=TASK/"mjbatch_intent_floor_inputs_v1/walk003/reference.npz"
TEACHER=TASK/"bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1"
KIND="native23_causal_context_target_student_v1"
FEATURES=1323

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def archive(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}

def finite_json(value):
    """Retain explicit failures while encoding nonfinite diagnostics as null."""
    if isinstance(value,dict):return {str(k):finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_json(v) for v in value]
    if isinstance(value,np.ndarray):return finite_json(value.tolist())
    if isinstance(value,np.generic):return finite_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value
