"""Prepare source only. This script performs no model, native or training calls."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
old = BASE.parent / "direct_target_full_state_evaluation_v1/source_draft_v1"
new = BASE / "source_draft_v1"
new.mkdir(exist_ok=False)
before = {}
for path in sorted(old.rglob("*.py")):
    relative = path.relative_to(old)
    dest = new / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw = path.read_bytes()
    dest.write_bytes(raw)
    before[relative.as_posix()] = hashlib.sha256(raw).hexdigest()
path = new / "direct_runtime.py"
text = path.read_text()
changes = [
    ("from direct_features import DirectFeatures", "from direct_features import DirectFeatures\nfrom causal_features import CausalFeatures"),
    ("head.get_inputs()[0].shape[-1]!=1000", "head.get_inputs()[0].shape[-1]!=1323"),
    ("direct features1000 ONNX input required", "causal features1323 ONNX input required"),
    ("cls(seed,head,DirectFeatures(motion,original29,c),c,span,timeline,infer_base)",
     "cls(seed,head,CausalFeatures(DirectFeatures(motion,original29,c),seed),c,span,timeline,infer_base)"),
    ("np.zeros(1000,np.float32) if self.current_mode==2", "np.zeros(1323,np.float32) if self.current_mode==2"),
]
for previous, replacement in changes:
    assert text.count(previous) == 1, previous
    text = text.replace(previous, replacement)
path.write_text(text, encoding="utf-8", newline="\n")
(BASE / "original_runtime_source_sha256.json").write_text(json.dumps(before, indent=2)+"\n")
print(json.dumps({"copied_sources": len(before), "runtime_only_changed": True,
                  "evaluation_gates_not_yet_adapted": True, "model_calls": 0, "native_calls": 0}))
