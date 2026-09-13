"""Freeze a future completed-run saved audit; no process/model/native calls."""
import argparse
from pathlib import Path
from audit_clock import local,read,write,sha


def prepare(clock_root,review_path,pass_field,selected):
    if selected is not True:raise ValueError('Explicit parent selection required')
    base=local(clock_root).resolve();folder=base/'clock_process';output=base/'run'
    source=Path(__file__).resolve().parent;source_hashes={p.name:sha(p) for p in source.glob('*.py')}
    review_path=local(review_path).resolve();review=read(review_path)
    if review.get(pass_field) is not True or review.get('source_sha256')!=source_hashes:
        raise ValueError('Literal current source review required')
    roles=dict(run_request=base/'clock_request.json',launch_receipt=folder/'launch_receipt.json',clearance=folder/'launch_clearance.json',
        run_report=output/'report.json',owner=base/'owner_completion.json',output_manifest=output/'output_manifest.json',
        start=folder/'start.json',child=folder/'child.json',raw_exit=folder/'raw_exit.json',exit=folder/'exit.json',
        diagnostic=folder/'diagnostic_verdict.json',absence=folder/'process_absence.json',launch_pre=folder/'prerun_hashes.json',launch_post=folder/'postrun_hashes.json')
    run=read(roles['run_request']);receipt=read(roles['launch_receipt']);owner=read(roles['owner']);clearance=read(roles['clearance'])
    if owner.get('accounting_passed') is not True:raise ValueError('Completed accounting evidence required; diagnostic failure may remain')
    roles['launch_review']=local(clearance['review']['path']).resolve()
    for key,original in [('fixture','canonical_fixture'),('expected_mjb','expected_model_mjb'),('expert_main','expert_main'),('expert_hold','expert_hold'),('reference','reference')]:
        roles[key]=local(run['roles'][original]).resolve()
    roles['contract']=local(run['native_bundle']).resolve()/'contract.json'
    roles['clock_runner']=local(run['source_directory']).resolve()/'run_clock.py'
    roles['clock_watchdog']=local(run['source_directory']).resolve()/'stage_watchdog.py'
    roles['clock_core']=local(run['source_directory']).resolve()/'clock_core.py'
    roles['clock_worker']=local(run['source_directory']).resolve()/'dummy_worker.py'
    roles['clock_protocol']=local(run['source_directory']).resolve()/'recorded_protocol.py'
    roles['clock_pending_result']=local(run['source_directory']).resolve()/'pending_result.py'
    roles['original_core']=source.parents[1]/'independent_plant_clock_timeout_correction_v1/source_draft_v1/clock_core.py'
    roles['original_runner']=source.parents[1]/'independent_plant_process_clock_v1/source_runner_v1/run_clock.py'
    import clock_stage_math
    prior=source.parents[1]/'independent_plant_pending_result_v1/source_draft_v1'
    for key in clock_stage_math.SOURCE_SHA:
        roles['prior_'+key]=roles[key] if key.startswith('original_') else prior/roles[key].name
    roles['timing_source_preparation']=source.parents[1]/'independent_plant_timing_integration_v1/source_preparation.json'
    roles['timing_source_review']=source.parents[1]/'independent_timing_integration_root_review_v1/review.json'
    stage_directory=local(run['stage_directory']).resolve()
    if stage_directory!=base/'stage_receipts':raise ValueError('Expected reviewed adjacent stage directory')
    stage_paths=sorted(stage_directory.glob('*.json'))
    if not stage_paths:raise ValueError('Completed durable stage evidence required')
    pins={}
    def bind(path,expected=None):
        path=local(path).resolve();actual=sha(path)
        if expected is not None and actual!=expected:raise ValueError('Changed completed input/output: '+str(path))
        if path.as_posix() in pins and pins[path.as_posix()]!=actual:raise ValueError('Conflicting pin')
        pins[path.as_posix()]=actual
    for path,digest in receipt['input_hashes'].items():bind(path,digest)
    for path,digest in owner['output_hashes'].items():bind(path,digest)
    for path in roles.values():bind(path)
    for path in stage_paths:
        key=path.as_posix()
        if key not in {local(k).resolve().as_posix() for k in owner['output_hashes']}:raise ValueError('Stage record missing from completed owner')
        bind(path)
    for p in source.glob('*.py'):bind(p)
    bind(review_path)
    for relative,entry in read(roles['output_manifest'])['files'].items():
        path=(output/relative).resolve()
        if not path.is_relative_to(output.resolve()):raise ValueError('Output escapes run directory')
        bind(path,entry['sha256'])
    return dict(kind='independent_saved_clock_audit',root_selected_saved_audit=True,numpy_version='1.26.4',
        stage_records=[{'path':p.as_posix(),'sha256':sha(p)} for p in stage_paths],
        roles={k:v.as_posix() for k,v in roles.items()},input_sha256=dict(sorted(pins.items())),source_sha256=source_hashes,
        source_review=dict(path=review_path.as_posix(),sha256=sha(review_path),pass_field=pass_field),
        native_steps=0,model_calls=0,optimizer_updates=0,worker_processes=0,
        outcome_contract='Evidence integrity is separate from physical, command, timing and component qualification.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--clock-root',type=Path,required=True);p.add_argument('--source-review',type=Path,required=True)
    p.add_argument('--pass-field',default='passed');p.add_argument('--root-selected',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=prepare(a.clock_root,a.source_review,a.pass_field,a.root_selected);write(local(a.output),result)


if __name__=='__main__':main()
