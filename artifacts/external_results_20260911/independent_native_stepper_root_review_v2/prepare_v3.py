from pathlib import Path
import hashlib
import json
B=Path(__file__).resolve().parent
BASE=B.parent/'independent_native_stepper_v1'
OLD=BASE/'source_draft_v2';NEW=BASE/'source_draft_v3'
NEW.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old_pins={p.name:sha(p) for p in OLD.glob('*.py')}
for p in OLD.glob('*.py'):(NEW/p.name).write_bytes(p.read_bytes())
p=NEW/'native_stepper.py';s=p.read_text()
old="if self.restore_attempted or self.returned or self.attempted:raise NativeAdapterError('initial restoration is allowed once; no rearm/reset')"
new="if self.closed or self.failure is not None or self.restore_attempted or self.returned or self.attempted:raise NativeAdapterError('initial restoration requires an open, unfailed adapter and is allowed once; no rearm/reset')"
assert s.count(old)==1
p.write_text(s.replace(old,new))
test='''"""Closed adapter must reject initialization before state writes; fake API only."""
import unittest
from test_native_stub import fixture
from native_stepper import NativeAdapterError

class ClosedRestoreTests(unittest.TestCase):
    def test_closed_uninitialized_adapter_rejects_restore_without_mutation(self):
        s,m,d,a=fixture(False)
        initial=d.integration.copy()
        s.verify_model_exit()
        with self.assertRaises(NativeAdapterError):
            s.restore_initial(initial,d.warning.number.copy(),d.warning.lastinfo.copy())
        self.assertTrue(s.closed)
        self.assertFalse(s.initialized)
        self.assertFalse(s.restore_attempted)
        self.assertEqual(a.set_calls,0)
        self.assertEqual(a.calls,[])
        self.assertEqual(d.integration.tobytes(),initial.tobytes())

if __name__=='__main__':unittest.main()
'''
(NEW/'test_closed_restore.py').write_text(test)
assert old_pins=={p.name:sha(p) for p in OLD.glob('*.py')}
report=dict(kind='root source-only closed-restore correction',old_source_pins=old_pins,
            new_source_pins={p.name:sha(p) for p in NEW.glob('*.py')},
            exact_change={'old':old,'new':new},real_model_constructions=0,real_native_steps=0)
(B/'v3_source_derivation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'source':str(NEW),'native_stepper_sha256':sha(p)}))
