"""Create one real saved-audit request only from completed stage artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
from audit_saved import local,read,sha

BASE=Path(__file__).resolve().parent
NEW=BASE.parent.parent
def subject(path):
    path=local(path).resolve()
    if not path.is_file():raise ValueError('Actual completed input missing: '+str(path))
    return dict(path=path.as_posix(),sha256=sha(path))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stages',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();config=read(a.stages)
    assert set(config)=={'witness','replay'}
    stages={}
    scalar_roles=('request','clearance','report','owner','process_exit','launch_receipt','process_absence','pre_hashes','post_hashes')
    for phase in ('witness','replay'):
        spec=config[phase];stages[phase]={key:subject(spec[key]) for key in scalar_roles}
        stages[phase]['completion_evidence']=[subject(p) for p in spec['completion_evidence']]
        owner=read(spec['owner']);report=read(spec['report']);exit_receipt=read(spec['process_exit'])
        assert owner['passed'] is True and report['passed'] is True and report['error'] is None and exit_receipt['exit_code']==0
        assert owner['request_sha256']==stages[phase]['request']['sha256'] and owner['report_sha256']==stages[phase]['report']['sha256']
        assert report['request_sha256']==stages[phase]['request']['sha256']
    native=read(config['replay']['request'])
    reports={
        'expert_main':NEW/'bfm250_expert_full_independent_physics_v1/report.json',
        'expert_hold':NEW/'bfm250_expert_hold_independent_physics_v1/report.json',
        'direct_failure':NEW/'direct_target_independent_physics_v1/report.json'}
    request=dict(kind='saved_native_stepper_equivalence_audit',completed_native_run_required=True,stages=stages,
        source_files=[subject(BASE/name) for name in ('audit_saved.py','saved_math.py','prepare_request.py','test_saved_audit.py')],
        stage_config=subject(a.stages),native_contract=subject(local(native['native_bundle'])/'contract.json'),
        canonical_fixture=subject(native['canonical_fixture']),qualified_traces={name:subject(path) for name,path in native['traces'].items()},
        qualification_reports={name:subject(path) for name,path in reports.items()},
        budgets=dict(model_calls=0,native_steps=0,optimizer_updates=0),expected_native_capture_rows=21348,
        actual_runtime_execution=False,hardware_authorized=False)
    with local(a.output).open('x',encoding='utf-8') as f:json.dump(request,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'saved_request':str(a.output),'sha256':sha(a.output)}))

if __name__=='__main__':main()
