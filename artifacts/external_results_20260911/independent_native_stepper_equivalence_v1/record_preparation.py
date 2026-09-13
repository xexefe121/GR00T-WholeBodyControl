"""Freeze source-only preparation receipts; no native imports or calls."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    files=[BASE/n for n in ('prepare_stage.py','stage_verdict.py','verify_completed_stage.py',
        'test_stage_preparation.py','stage_stub_tests_v3.xml','README.md','proposal.json',
        'source_derivation.json','stub_tests_v3.xml','launcher_template_syntax_v1/report.json','record_preparation.py')]
    review=BASE.parent/'independent_native_stepper_equivalence_root_review_v1/review.json'
    files.append(review);r=read(review);assert r['source_review_passed'] is True
    counts=[]
    for name in ('stub_tests_v3.xml','stage_stub_tests_v3.xml'):
        root=ET.parse(BASE/name).getroot();suites=list(root.iter('testsuite'))
        assert all(int(s.get('failures','0'))==int(s.get('errors','0'))==int(s.get('skipped','0'))==0 for s in suites)
        counts.append(sum(int(s.get('tests','0')) for s in suites))
    assert counts==[26,15]
    syntax=read(BASE/'launcher_template_syntax_v1/report.json');assert syntax['passed'] is True and syntax['scripts']==4
    files.extend(p for p in (BASE/'launcher_template_syntax_v1').glob('*.ps1'))
    result=dict(kind='native_adapter_equivalence_source_preparation',source_preparation_only=True,
        actual_native_steps=0,actual_serializations=0,model_inference_calls=0,optimizer_updates=0,
        source_root_review_sha256=sha(review),proposal_sha256=sha(BASE/'proposal.json'),
        synthetic_tests=41,input_hashes={p.resolve().as_posix():sha(p) for p in files},
        planned_witness_steps=0,planned_witness_serializations=2,planned_replay_steps=21348,
        planned_replay_serializations=8,concrete_final_review_pending=True)
    out=BASE/'source_preparation.json'
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(path=str(out),sha256=sha(out),tests=41)))
if __name__=='__main__':main()
