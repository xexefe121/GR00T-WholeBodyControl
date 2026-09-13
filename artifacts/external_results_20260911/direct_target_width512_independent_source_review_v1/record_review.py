"""Freeze source review and synthetic evidence; never imports task models."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
ORIGINAL=NEW/'direct_target_causal_width512_preparation_v1'
SOURCE=ORIGINAL/'source_prepared_v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
prep_path=ORIGINAL/'source_preparation.json'
assert sha(prep_path)=='33c12592a71a709b78e0843219541e6cfc9124ba25027ff103820c77c7617ac2'
prep=read(prep_path);assert prep['source_preparation_passed'] is True
source_map={p.name:sha(p) for p in SOURCE.iterdir() if p.is_file()}
assert source_map==prep['source_sha256'] and len(source_map)==3
inputs={prep_path.as_posix():sha(prep_path)}
for name,digest in source_map.items():inputs[(SOURCE/name).as_posix()]=digest
for path,digest in prep['input_sha256'].items():
    assert sha(path)==digest;inputs[Path(path).resolve().as_posix()]=digest
proposal=ORIGINAL/'proposal.json';assert sha(proposal)==prep['proposal_sha256'];inputs[proposal.as_posix()]=sha(proposal)
q=read(proposal)
assert q['actual_fit_selected'] is False and q['updates']==10000 and q['architecture']==[1323,512,512,23]
assert q['parity_tolerance_rad']==1e-5 and q['training_rows']==146860000
runtime=read(BASE/'runtime.json');test_exit=read(BASE/'test_exit.json')
assert runtime['torch_threads']==runtime['interop_threads']==1 and runtime['cuda_initialized'] is False
assert test_exit['exit_code']==0 and test_exit['cuda_initialized'] is False
xml=ET.parse(BASE/'synthetic_tests.xml').getroot();suites=[xml] if xml.tag=='testsuite' else list(xml)
assert sum(int(s.attrib['tests']) for s in suites)==27
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
for name in ('run_synthetics.py','record_review.py','NOTE.md','runtime.json','test_exit.json','synthetic_tests.xml','synthetic_tests.log'):
    inputs[(BASE/name).as_posix()]=sha(BASE/name)
review=dict(passed=True,source_review_pass=True,preparation_only=True,reviewer='pico_continuation',
    source_preparation_subject={'path':prep_path.as_posix(),'sha256':sha(prep_path)},source_sha256=source_map,
    source_directory=SOURCE.as_posix(),proposal_sha256=sha(proposal),input_sha256=inputs,
    independent_synthetic_tests=27,tests_failed=0,tests_skipped=0,runtime=runtime,findings=[],
    checked=['Original 1000/323 split and 256-wide old contractions preserve initialization function.',
        'Added incoming randomness and zero outgoing blocks preserve source while enabling later gradient flow.',
        'Signed-zero custom addition retains both real-addition gradients and higher-order differentiation.',
        'All six AdamW old blocks, moments, counters and group fields preserved; added moments zero.',
        'Separate local expansion RNG and copied restoration payload do not consume global CPU RNG.',
        'Fixed schedule endpoints and row budget agree; final FP64 numerical/native gates remain required.'],
    limitations=['CPU synthetic evidence only; actual CUDA checkpoint arithmetic and end-to-end timing unqualified.',
        'Shared step 6000 gives added zero moments warm bias correction, not a fresh optimizer trajectory.',
        'Width and learning-rate ramp are simultaneous engineering changes; no capacity-causation claim.',
        'Actual RNG restoration, exporter, final parity and full native acceptance remain future gates.'],
    actual_checkpoint_loads=0,task_arrays_loaded=0,actual_task_model_calls=0,actual_task_optimizer_updates=0,native_steps=0,
    synthetic_torch_forwards_backwards_and_optimizer_step_performed=True,cuda_initialized=False,
    fit_created=False,actual_training_selected=False,execution_clearance=False)
out=BASE/'review.json'
with out.open('x',encoding='utf-8') as f:json.dump(review,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,review_sha256=sha(out),source_sha256=source_map,tests=27)))
