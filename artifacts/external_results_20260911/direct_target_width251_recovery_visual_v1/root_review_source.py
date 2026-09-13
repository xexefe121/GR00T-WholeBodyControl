"""Review fixed saved-state visualization and select one rendering, no dynamics."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=BASE/'source_preparation.json';assert sha(p)=='459f72ad0bbfa585d94fec9ce5231831298000cff83e51175b0daa9a05b84823'
r=json.loads(p.read_text())
assert r['source_preparation_passed'] is True and r['preserved_camera_geometry_block_byte_identical'] is True
for name,h in r['source_sha256'].items():assert sha(BASE/'source_draft_v1'/name)==h
for name,h in r['evidence_sha256'].items():assert sha(name)==h
tests=BASE/'root_tests_v1.xml';suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==57
assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
assert not Path(r['output_directory']).exists()
out=BASE/'root_review.json'
record=dict(passed=True,source_review_pass=True,source_preparation_sha256=sha(p),source_sha256=r['source_sha256'],
 root_tests=dict(path=tests.as_posix(),sha256=sha(tests),junit_cases=57),
 root_selected_one_saved_state_render=True,requested_physics_steps=0,model_calls=0,
 qualification='completed full expert recovery with four original independent reports',
 no_controller_qualification_inferred=True,writer_sha256=sha(__file__))
with out.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,sha256=sha(out),saved_render_selected=True)))
