"""Independent pending-result source/AST and67 synthetic tests; no actual workers or native calls."""
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
TARGET=NEW/'independent_plant_pending_result_v1'
SOURCE=TARGET/'source_draft_v1'
ORIGINAL=NEW/'independent_plant_pending_publication_v1/source_draft_v1'
PREP=TARGET/'source_preparation.json'
EXPECTED='88080d6c84025b7589a381a1d5d3dc5f170acdb69b20c95f31123f3c6278b4ad'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as stream:json.dump(v,stream,indent=2,allow_nan=False);stream.write('\n')
def members(path):
    result={}
    def visit(nodes,prefix=''):
        for node in nodes:
            if isinstance(node,ast.ClassDef):visit(node.body,prefix+node.name+'.')
            elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                result[prefix+node.name]=ast.dump(node,include_attributes=False)
    visit(ast.parse(Path(path).read_text(encoding='utf-8-sig')).body)
    return result
def main():
    assert sha(PREP)==EXPECTED
    prep=read(PREP);assert prep['source_preparation_passed'] is True and prep['execution_selected'] is False
    pins={PREP.as_posix():EXPECTED};current={p.name:sha(p) for p in sorted(SOURCE.glob('*.py'))}
    old={p.name:sha(p) for p in sorted(ORIGINAL.glob('*.py'))}
    assert current==prep['source_sha256'] and old==prep['original_source_sha256']
    same=sorted(n for n in old if current[n]==old[n]);changed=sorted(n for n in old if current[n]!=old[n]);added=sorted(set(current)-set(old))
    assert len(current)==24 and len(same)==16 and changed==sorted(prep['changed_files']) and added==['pending_result.py','test_pending_result.py']
    assert same==sorted(prep['unchanged_files'])
    for folder,mapping in [(SOURCE,current),(ORIGINAL,old)]:
        for name,digest in mapping.items():
            pins[(folder/name).as_posix()]=digest
            ast.parse((folder/name).read_text(encoding='utf-8-sig'))
    diff=''.join(''.join(difflib.unified_diff((ORIGINAL/n).read_text().splitlines(True),(SOURCE/n).read_text().splitlines(True),
        fromfile='original/'+n,tofile='source_draft_v1/'+n)) for n in prep['changed_files'])
    assert diff==(TARGET/'source_changes.diff').read_text()
    allowed={'clock_core.py':{'Job.identity','PlantFoundation._admit','PlantFoundation._boundary'},
        'recorded_protocol.py':{'decode_job'},'dummy_worker.py':{'worker_once','process_main'},'run_clock.py':{'imported_source_paths'}}
    ast_checks={}
    for name,wanted in allowed.items():
        before,after=members(ORIGINAL/name),members(SOURCE/name)
        differing={n for n,v in before.items() if after.get(n)!=v}
        assert differing==wanted
        extra=set(after)-set(before)
        assert extra==({'Job.__post_init__'} if name=='clock_core.py' else set())
        ast_checks[name]=dict(changed=sorted(differing),new=sorted(extra),unchanged=sorted(set(before)-differing))
    critical=('PlantFoundation.tick','PlantFoundation.deadline','PlantFoundation.debt','PlantFoundation.epoch_ns',
        'PlantFoundation._validate_command','PlantFoundation.poll_results','PlantFoundation._attempt_job_publication',
        'PlantFoundation._retry_pending_publication','PlantFoundation._expire_pending_publication')
    for name in critical:assert name in ast_checks['clock_core.py']['unchanged']
    for name,digest in prep['evidence_sha256'].items():assert sha(TARGET/name)==digest;pins[(TARGET/name).as_posix()]=digest
    for item in prep['external_subjects'].values():assert sha(item['path'])==item['sha256'];pins[Path(item['path']).as_posix()]=item['sha256']
    write(BASE/'pretest_source_checks.json',dict(passed=True,source_sha256=current,input_sha256=pins,ast_checks=ast_checks,
        exact_declared_text_diff=True,unchanged_critical_plant_methods=list(critical)))
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):env[key]='1'
    env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=str(SOURCE)
    args=[sys.executable,'-m','unittest','discover','-s',str(SOURCE),'-p','test_*.py','-v']
    with (BASE/'synthetic_tests.log').open('xb') as stream:
        run=subprocess.run(args,cwd=SOURCE,env=env,stdout=stream,stderr=subprocess.STDOUT,check=False)
    log=(BASE/'synthetic_tests.log').read_text(encoding='utf-8-sig')
    write(BASE/'test_exit.json',dict(exit_code=run.returncode,arguments=args,runtime=sys.version,
        actual_worker_calls=0,actual_clock_runs=0,model_calls=0,native_steps=0))
    assert run.returncode==0 and 'Ran 67 tests' in log and log.rstrip().endswith('OK')
    for path,digest in pins.items():assert sha(path)==digest,path
    result=dict(passed=True,source_review_pass=True,preparation_only=True,
        source_preparation_subject=dict(path=PREP.as_posix(),sha256=EXPECTED),source_sha256=current,
        input_sha256=pins,ast_checks=ast_checks,unchanged_files=same,changed_files=changed,new_files=added,
        exact_declared_text_diff=True,independent_synthetic_tests=67,test_exit_sha256=sha(BASE/'test_exit.json'),
        tests_sha256=sha(BASE/'synthetic_tests.log'),review_source_sha256=sha(__file__),all_source_pins_rehashed=True,
        findings=[],reviewed_contracts=[
            'Original plant epoch+activation deadline travels in canonical Job and Result identity; plant recomputes before admission.',
            'One reply payload computed and retained; only fully recorded BUSY permits another nonblocking publication.',
            'One attempt per worker iteration,20 total, no attempt starts at or after original deadline; late published returns remain failures.',
            'No new polls while pending/buffered; fixed two taken-job buffer retains immutable payloads and rejects duplicates.',
            'Attempt/return/ready/publication/terminal counters commit separately; post-write exceptions, invalid clocks and logging failures preserve ambiguous evidence without retry.',
            'Plant tick/deadline/debt/native/PD/history/capture/mailbox and worker lifecycle remain exact; only explicit identity/decode/worker retry changes.'
        ],actual_process_calls=0,actual_worker_calls=0,actual_clock_runs=0,model_calls=0,ORT_calls=0,native_steps=0,optimizer_updates=0,
        execution_selected=False,actual_timing_qualified=False,physical_qualified=False,
        limitations=['Synthetic/source qualification only; matching saved-audit schema and concrete selection required before actual clock run.',
            'Additional allocation/logging may affect timing; fixed ledger overflow, deadline expiry or hard-kill evidence loss remain failures.',
            'No inference about recovery of prior15 timing misses or full-scope physical success.'])
    write(BASE/'review.json',result)
    print(json.dumps(dict(passed=True,tests=67,review_sha256=sha(BASE/'review.json'))))
if __name__=='__main__':main()
