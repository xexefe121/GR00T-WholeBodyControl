"""Freeze reviewed-text candidate and synthetic tests, never task state."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_causal_width512_student_v1/source_prepared_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    draft=BASE/'source_draft_v1';snapshot=BASE/'source_prepared_v1'
    copies=['width512.py','restoration_support.py','balance_contract.py']
    for name in copies:
        if (draft/name).read_bytes()!=(OLD/name).read_bytes():raise ValueError('Changed prior copy: '+name)
    suites=list(ET.parse(BASE/'synthetic_tests_v1.xml').getroot().iter('testsuite'))
    counts={key:sum(int(s.get(key,'0')) for s in suites) for key in ('tests','failures','errors','skipped')}
    if counts!={'tests':43,'failures':0,'errors':0,'skipped':0}:raise ValueError('Expected43 passing CPU synthetic tests')
    snapshot.mkdir()
    for p in sorted(draft.glob('*.py')):shutil.copyfile(p,snapshot/p.name)
    report=dict(source_preparation_pass=True,prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_directory=snapshot.as_posix(),source_sha256={p.name:sha(p) for p in sorted(snapshot.glob('*.py'))},
        unchanged_sources={name:dict(path=(OLD/name).as_posix(),sha256=sha(OLD/name)) for name in copies},
        source_references={p.as_posix():sha(p) for p in [OLD/'train_response_balanced.py',OLD/'response_contract.py',OLD/'training_support.py']},
        tests=dict(path=(BASE/'synthetic_tests_v1.xml').as_posix(),sha256=sha(BASE/'synthetic_tests_v1.xml'),**counts),
        design_sha256=sha(BASE/'DESIGN.md'),preparation_source_sha256=sha(__file__),
        scope=dict(source_ordinary_step=81000,source_optimizer_step=16000,architecture=[1323,512,512,23],
            actor_tensors=6,optimizer_states=6,expansion=False,normalization_refit=False,zero_reset=False),
        actual_checkpoint_reads=0,actual_task_data_reads=0,model_forward_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,
        root_source_review_required=True,actual_fit_selected=False,update_count_selected=False,learning_rate_selected=False,recovery_coefficient_selected=False)
    target=BASE/'source_preparation.json'
    with target.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(path=target.as_posix(),sha256=sha(target),sources=len(report['source_sha256']),tests=counts)))
if __name__=='__main__':main()
