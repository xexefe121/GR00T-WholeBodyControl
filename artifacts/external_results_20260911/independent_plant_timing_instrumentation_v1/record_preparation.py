"""Freeze the sidecar proposal and fake-only test evidence; do not integrate it."""
import hashlib,json,ast
from pathlib import Path
BASE=Path(__file__).resolve().parent;NEW=BASE.parent;SOURCE=BASE/'source_draft_v1'
PRODUCER=NEW/'independent_plant_pending_result_v1/source_draft_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,j):
    with Path(p).open('x') as f:json.dump(j,f,indent=2);f.write('\n')
def main():
    prep_path=PRODUCER.parent/'source_preparation.json';prep=read(prep_path)
    assert sha(prep_path)=='88080d6c84025b7589a381a1d5d3dc5f170acdb69b20c95f31123f3c6278b4ad'
    for name,h in prep['source_sha256'].items():assert sha(PRODUCER/name)==h
    log=(BASE/'synthetic_tests_v2.log').read_text();assert 'Ran 22 tests' in log and log.rstrip().endswith('OK')
    proposed=[('session.py','Session.tick_once',['TICK_ENVELOPE','OUTER_LOG']),
        ('clock_core.py','PlantFoundation.tick',['RESULT_POLL_AND_ADMIT','FIXED_WAIT','CAPTURE_OWNERSHIP','VERIFY_OWNERSHIP','STEP_LEDGER']),
        ('clock_core.py','PlantFoundation._boundary',['BOUNDARY_SNAPSHOT','BOUNDARY_HISTORY','BOUNDARY_SERIALIZATION']),
        ('clock_core.py','PlantFoundation._attempt_job_publication',['JOB_PUBLICATION']),
        ('native_stepper.py','NativeStepper.step',['PD_AND_INPUT','MJ_STEP']),
        ('clock_core.py','PlantFoundation.tick',['CAPTURE','VERIFY'])]
    sites=[]
    for name,qualified,phases in proposed:
        path=PRODUCER/name;tree=ast.parse(path.read_text());klass,method=qualified.split('.')
        node=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name==klass for n in c.body if isinstance(n,ast.FunctionDef) and n.name==method)
        sites.append(dict(file=path.as_posix(),source_sha256=sha(path),function=qualified,line=node.lineno,phases=phases,
            original_function_ast_sha256=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),inserted=False))
    write(BASE/'hook_plan.json',dict(proposal_only=True,hook_sites=sites,producer_source_sha256=prep['source_sha256'],
        source_runtime_modified=False,full_controls=1819,full_native_steps=18190,model_serializations=4,
        span_capacity=262144,planned_span_bound=223727,gc_event_capacity=4096,table_bytes=33947648,
        fixed_step_ns=2000000,fixed_control_ns=20000000,callback_registry_changed=False))
    inputs={prep_path.as_posix():sha(prep_path)}
    for name,h in prep['source_sha256'].items():inputs[(PRODUCER/name).as_posix()]=h
    for relative in ('independent_pending_result_clock_diagnosis_v1/report.json',
        'independent_pending_result_clock_diagnosis_v1/diagnostic_receipt.json',
        'independent_plant_pending_result_saved_actual_v1/results_v1/report.json',
        'independent_plant_pending_result_saved_actual_v1/owner_completion.json'):
        path=NEW/relative;inputs[path.as_posix()]=sha(path)
    report=dict(passed=True,source_preparation_only=True,source_sha256={p.name:sha(p) for p in SOURCE.glob('*.py')},
        evidence_sha256={name:sha(BASE/name) for name in ('DESIGN.md','hook_plan.json','synthetic_tests_v1.log','synthetic_tests_v2.log','record_preparation.py')},
        tests=22,failures=0,errors=0,skips=0,input_sha256=inputs,
        actual_clock_readers_used=False,actual_gc_callbacks_installed=0,native_steps=0,model_calls=0,optimizer_updates=0,
        worker_processes_started=0,runtime_source_changes=0,actual_request_created=False,actual_run_selected=False,
        limitations=['Not integrated into the native/runtime producer; exact hook insertion and saved-audit reconciliation remain source work.',
            'Fixed preallocated storage does not eliminate Python allocation or measurement overhead.',
            'CPU/GC overlap evidence localizes observations but does not establish a stall cause or timing/balance qualification.'])
    write(BASE/'source_preparation.json',report)
    print(json.dumps(dict(source_preparation_sha256=sha(BASE/'source_preparation.json'),source_sha256=report['source_sha256'],tests=22)))
if __name__=='__main__':main()
