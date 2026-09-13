"""Injected stub regression only; zero native model/physics calls."""
from pathlib import Path
import sys
import json
import hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'independent_native_stepper_v1/source_draft_v2'))
from test_native_stub import fixture
from native_stepper import NativeAdapterError
s,m,d,a=fixture(False)
s.verify_model_exit()
error=None
try:s.restore_initial(d.integration.copy(),d.warning.number.copy(),d.warning.lastinfo.copy())
except NativeAdapterError as exc:error=str(exc)
report=dict(kind='closed adapter initial-restore stub regression',closed=s.closed,error=error,
            rejected_before_any_state_write=error is not None and a.set_calls==0,
            fake_set_calls=a.set_calls,fake_calls=a.calls,
            real_model_constructions=0,real_native_steps=0,
            source_sha256=hashlib.sha256(Path(sys.modules['native_stepper'].__file__).read_bytes()).hexdigest())
out=Path(__file__).with_name('closed_restore_v2.json')
assert not out.exists()
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
