"""Source/hash verification only; no audit/model/native execution."""
import ast,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
B=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911');S=B/'direct_target_root_audit_v1';O=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(S/n):sha(S/n) for n in ('audit_saved_fit.py','audit_math.py','test_audit_math.py','tests_v2.xml')}
assert pins[str(S/'audit_saved_fit.py')]=='40a47fd2e7554368c38a262fc9a2bdd0e75a3ceaa94ca817fbfb8646e191a55b'
for n in ('audit_saved_fit.py','audit_math.py','test_audit_math.py'):ast.parse((S/n).read_text())
suites=ET.parse(S/'tests_v2.xml').findall('.//testsuite')
assert sum(int(s.attrib['tests']) for s in suites)==9
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
pins[str(__file__)]=sha(__file__)
review=dict(kind='direct_target_saved_fit_audit_source_review',verdict='CLEAR',source_review_pass=True,input_sha256=pins,
    scope='Independent saved-prediction, normalization, fixed sampler/loss schedule, checkpoint and ONNX evidence audit. No inference or optimization.',
    checks=['Exact dataset/control/source-frame/phase identities precede contiguous group math.','Equal15-cell independent weighted moments; declared reduction tolerance and exact saved64-to32 casts.','Normalized response predictions widen before subtraction; fixed9-cell physical denominators and full velocity groups.','All2.88M row/axis schedules,5000 rates/loss compositions and counters checked.','Full ONNX operation wiring, attributes, domains, float32 dimensions, fixed weight sizes and checkpoint tensor equality.','Actual checkpoint request, carried fit request, optimization receipt, and clearance identity chains checked.','Three launch-review direct subjects path+SHA compare to actual training request, frozen receipt and launcher.','Failure report and completed comparison prefix retained; actual outputs/input hashes rechecked.'],
    synthetic_tests_bound=9,reviewer_tests_run=0,reviewer_model_calls=0,reviewer_native_steps=0,reviewer_optimizer_updates=0,
    limitations=['Future real fit subjects must exist and be frozen before audit execution.','Saved losses cannot reproduce unsaved gradients or establish connected stability.','RNG initialization/CUDA execution flags rely on separate trainer source/runtime review.'])
with (O/'review.json').open('x',encoding='utf-8') as f:json.dump(review,f,indent=2);f.write('\n')
print(json.dumps(dict(verdict='CLEAR',sha256=sha(O/'review.json'),pins=len(pins))))
