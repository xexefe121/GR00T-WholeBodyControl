"""Failure-aware independent saved-clock audit. Never calls native/model/worker APIs."""
import argparse,hashlib,json,sys,traceback
from pathlib import Path
import numpy as np
import clock_saved_math as physical
import clock_control_math as control
import clock_protocol_math as protocol
import clock_accounting as accounting
import clock_stage_math as stage_math


def local(value):
    text=str(value).replace('\\','/')
    if sys.platform!='win32' and len(text)>2 and text[1]==':':text='/mnt/'+text[0].lower()+text[2:]
    return Path(text)
def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def archive(path):
    with np.load(path,allow_pickle=False) as z:return {key:z[key].copy() for key in z.files}


class Proof:
    def __init__(self,output):
        self.output=output;self.count=0;self.stage='input gate';self.first_failure=None
        self.stream=(output/'comparisons.jsonl').open('x',encoding='utf-8')
    def note(self,name,passed=True):
        self.count+=1;self.stream.write(json.dumps({'check':self.count,'stage':self.stage,'name':name,'passed':passed})+'\n')
    def exact(self,actual,expected,name):
        a,b=np.asarray(actual),np.asarray(expected)
        passed=a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes();self.note(name,passed)
        if not passed:
            if self.first_failure is None:
                self.first_failure={'name':name,'stage':self.stage,'check':self.count}
                np.savez(self.output/'first_mismatch.npz',actual=a,expected=b)
            raise AssertionError(name+': independent dtype/shape/byte mismatch')


def validate_request(request_path):
    request=read(request_path)
    if request['kind']!='independent_saved_clock_audit' or request['root_selected_saved_audit'] is not True:raise ValueError('Saved audit selection required')
    if np.__version__!=request['numpy_version'] or request['numpy_version']!='1.26.4':raise ValueError('Pinned saved-audit NumPy1.26.4 required')
    files={local(k).resolve():v for k,v in request['input_sha256'].items()}
    if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('Frozen audit input changed')
    source=Path(__file__).resolve().parent
    current={p.name:sha(p) for p in source.glob('*.py')}
    if current!=request['source_sha256']:raise ValueError('Auditor source set changed')
    if any(files.get((source/name).resolve())!=digest for name,digest in current.items()):raise ValueError('Auditor final-rehash source pin missing')
    review_entry=request['source_review'];review_path=local(review_entry['path']).resolve()
    if files.get(review_path)!=review_entry['sha256']:raise ValueError('Unpinned source review')
    review=read(review_path)
    if review[review_entry['pass_field']] is not True or review['source_sha256']!=current:raise ValueError('Actual source review differs')
    roles={k:local(v).resolve() for k,v in request['roles'].items()}
    if any(path not in files for path in roles.values()):raise ValueError('Consumed role not pinned')
    return request,files,roles


def run_audit(request,roles,proof):
    for module in (physical,control,protocol,accounting):module.exact=proof.exact
    run_request=read(roles['run_request']);receipt=read(roles['launch_receipt']);clearance=read(roles['clearance'])
    report=read(roles['run_report']);owner=read(roles['owner']);manifest=read(roles['output_manifest'])
    pinned={local(k).resolve():v for k,v in request['input_sha256'].items()}
    for path,digest in receipt['input_hashes'].items():
        if pinned.get(local(path).resolve())!=digest:raise AssertionError('Actual launch input not frozen in saved audit')
    request_sha=sha(roles['run_request']);launch_sha=sha(roles['launch_receipt'])
    for role,original in [('fixture','canonical_fixture'),('expected_mjb','expected_model_mjb'),('expert_main','expert_main'),('expert_hold','expert_hold'),('reference','reference')]:
        if roles[role]!=local(run_request['roles'][original]).resolve():raise AssertionError('Audit role does not match actually consumed input: '+role)
    if roles['contract']!=local(run_request['native_bundle']).resolve()/'contract.json':raise AssertionError('Actual native contract role differs')
    if not (request_sha==receipt['request_sha256']==clearance['request_sha256']==report['request_sha256']==owner['request_sha256']==manifest['request_sha256']):raise AssertionError('Original request subject mismatch')
    if clearance['root_selected_single_run'] is not True or clearance['launch_receipt_sha256']!=launch_sha:raise AssertionError('Actual launch selection missing')
    launch_review=read(roles['launch_review']);entry=clearance['review']
    if entry['sha256']!=sha(roles['launch_review']) or launch_review[entry['pass_field']] is not True:raise AssertionError('Launch review differs')
    for key,path in [('request_subject',roles['run_request']),('launch_receipt_subject',roles['launch_receipt'])]:
        if local(launch_review[key]['path']).resolve()!=path or launch_review[key]['sha256']!=sha(path):raise AssertionError('Literal launch review subject differs')
    output=roles['run_report'].parent;all_output={}
    for relative,record in manifest['files'].items():
        path=(output/relative).resolve()
        if not path.is_relative_to(output) or sha(path)!=record['sha256'] or path.stat().st_size!=record['bytes']:raise AssertionError('Output manifest changed')
        all_output[path]=record['sha256']
    for path,digest in owner['output_hashes'].items():
        if sha(local(path))!=digest:raise AssertionError('Owner output changed')
    for name,path in [('report_sha256',roles['run_report']),('process_exit_sha256',roles['exit']),
                      ('launch_receipt_sha256',roles['launch_receipt']),('clearance_sha256',roles['clearance']),('process_absence_sha256',roles['absence'])]:
        if owner[name]!=sha(path):raise AssertionError('Owner direct subject differs: '+name)
    process=accounting.owner_checks(owner,receipt,clearance,*[read(roles[name]) for name in ('start','child','raw_exit','exit','diagnostic','absence','launch_pre','launch_post')],report)
    start=read(roles['start'])
    if start['receipt_sha256']!=launch_sha or start['clearance_sha256']!=sha(roles['clearance']) or start['review_sha256']!=sha(roles['launch_review']):
        raise AssertionError('Actual wrapper start did not bind reviewed launch')
    absence=read(roles['absence']);linux_expected=[]
    if (output/'process_start.json').exists():
        started=read(output/'process_start.json');linux_expected.append(started['pid'])
        if started['request_sha256']!=request_sha:raise AssertionError('Linux native process request differs')
    if (output/'worker_ready.json').exists():
        ready=read(output/'worker_ready.json');linux_expected.append(ready['worker_pid'])
        if ready['resource_tracker_pid'] is not None:linux_expected.append(ready['resource_tracker_pid'])
        if ready['parent_pid']!=started['pid'] or ready['worker_pid']!=report['worker_pid']:raise AssertionError('Worker process lineage differs')
    if absence['linux_expected_pids']!=sorted(set(linux_expected)):raise AssertionError('Actual Linux PIDs not checked absent')
    # Hash evidence is transitive; every actually read output must also be in the frozen audit request.
    if not set(all_output)<=set(pinned) or any(pinned.get(local(k).resolve())!=v for k,v in owner['output_hashes'].items()):
        raise AssertionError('Consumed output not frozen')
    for key in ('requested_controls','main_controls','hold_controls','native_step_budget','serialization_budget'):
        if run_request[key]!={'requested_controls':1819,'main_controls':1569,'hold_controls':250,'native_step_budget':18190,'serialization_budget':4}[key]:raise AssertionError('Original benchmark scope changed')
    stage_math.request_contract(run_request,receipt['exact_wsl_arguments'])
    stage_source=stage_math.source_contract({key:roles[key] for key in stage_math.SOURCE_SHA})
    source_directory=local(run_request['source_directory']).resolve()
    if roles['clock_runner']!=source_directory/'run_clock.py' or roles['clock_watchdog']!=source_directory/'stage_watchdog.py' or roles['clock_core']!=source_directory/'clock_core.py':raise AssertionError('Actual clock source namespace differs')
    args=receipt['exact_wsl_arguments']
    runners=[local(v).resolve() for v in args if v.endswith('/run_clock.py')]
    pythonpaths=[local(v.split('=',1)[1]).resolve() for v in args if v.startswith('PYTHONPATH=')]
    if runners!=[roles['clock_runner']] or pythonpaths!=[source_directory]:raise AssertionError('Launched source namespace differs')
    stage_directory=local(run_request['stage_directory']).resolve()
    if stage_directory!=output.parent/'stage_receipts':raise AssertionError('Stage output namespace differs')
    stage_paths=[local(v['path']).resolve() for v in request['stage_records']]
    if stage_paths!=sorted(stage_directory.glob('*.json')):raise AssertionError('Missing or additional durable stage records')
    stage_records=[]
    owner_paths={local(k).resolve():v for k,v in owner['output_hashes'].items()}
    for i,(path,entry) in enumerate(zip(stage_paths,request['stage_records'])):
        record=read(path)
        if path.name!=str(i).zfill(2)+'_'+record['stage']+'.json':raise AssertionError('Stage filename/order differs')
        if pinned.get(path)!=entry['sha256'] or owner_paths.get(path)!=entry['sha256'] or sha(path)!=entry['sha256']:raise AssertionError('Durable stage record not bound by actual owner/audit')
        stage_records.append(record)
    proof.note('actual request/review/owner/process/output lineage exact')
    api=report['api_counters'];serializations=[]
    if api is not None:
        if not 0<=api['serialization_returned']<=api['serialization_attempted']<=4:raise AssertionError('MJB call count exceeds scope')
        records=api['serialization_records']
        if len(records)!=api['serialization_attempted']:raise AssertionError('Missing actual serialization output evidence')
        for i,record in enumerate(records):
            if record['index']!=i:raise AssertionError('Serialization index skipped')
            path=output/'mjb'/record['file'];expected=roles['expected_mjb']
            if sha(path)!=record['sha256'] or path.stat().st_size!=record['bytes']:raise AssertionError('Captured actual MJB buffer changed')
            equal=path.stat().st_size==expected.stat().st_size
            with path.open('rb') as a,expected.open('rb') as b:
                while equal:
                    av=a.read(1024*1024);bv=b.read(1024*1024)
                    if av!=bv:equal=False
                    if not av and not bv:break
            serializations.append({'index':i,'native_returned':record['native_returned'],'byteexact_expected_model':equal,'capture_error':record.get('capture_error')})
        mjb_pass=api['serialization_attempted']==api['serialization_returned']==4 and all(v['byteexact_expected_model'] and v['native_returned'] and not v['capture_error'] for v in serializations)
    else:mjb_pass=False
    if report['all_four_MJB_exact'] is not mjb_pass:raise AssertionError('Four actual MJB verdict differs')
    if report['session'] is None:
        if api is not None and api['step_attempted']!=0:raise AssertionError('Missing session despite native execution')
        if report['first_error'] is None:raise AssertionError('Missing setup failure')
        stage_result=stage_math.stages(stage_records,report,read(roles['raw_exit']),request_sha,sha(roles['run_report']),sha(roles['output_manifest']))
        return dict(stage_watchdog=stage_result,stage_sources=stage_source,evidence_integrity_passed=True,physical_pass=False,timing_pass=False,command_pass=False,component_qualified=False,
                    actual_native_steps=0,process=process,MJB_serializations=serializations,setup_failure=report['first_error'])
    proof.stage='actual captured physics and control'
    a=archive(output/'trace.npz');metadata=control.decode(read(output/'evidence.json'))
    capsule_index=read(output/'raw_capsules/index.json');capsules={}
    for name,entry in capsule_index.items():
        path=(output/'raw_capsules'/entry['path']).resolve()
        if not path.is_relative_to((output/'raw_capsules').resolve()) or sha(path)!=entry['sha256']:raise AssertionError('Raw capsule binding differs')
        raw=path.read_bytes()
        if len(raw)!=entry['bytes']:raise AssertionError('Raw capsule size differs')
        capsules[name]=raw
    initial=np.frombuffer(capsules['initial_capture_state'],np.float64).copy()
    if initial.shape!=(373,):raise AssertionError('Initial373 capture schema')
    fixture=archive(roles['fixture']);full=archive(roles['expert_main']);hold=archive(roles['expert_hold'])
    proof.exact(initial[:291],fixture['state_vector'],'Initial actual full291 equals canonical fixture')
    proof.exact(initial[:291],full['initial_integration'],'Initial actual full291 equals expert')
    proof.exact(initial[1:31],initial[291:321],'Initial full291/qpos overlap')
    proof.exact(initial[31:60],initial[321:350],'Initial full291/qvel overlap')
    proof.exact(np.frombuffer(capsules['initial_capture_torque'],np.float64),initial[89:112],'Initial ctrl capture')
    warnings=metadata['initial_warning_ledger']
    proof.exact(np.asarray(warnings['counts'],np.int32),full['physics_warning_number'][0],'Initial actual warnings')
    proof.exact(np.asarray(warnings['lastinfo'],np.int32),full['physics_warning_lastinfo'][0],'Initial warning lastinfo')
    contract={key:np.asarray(value,np.float64) for key,value in read(roles['contract']).items() if key in ('kp','kd','native_effort','native_velocity','joint_limits','default_q')}
    initial_result=physical.initial_strict(initial,np.asarray(warnings['counts']),contract)
    physical_result=physical.physical_arrays(a,initial[:291],contract)
    counter_result=accounting.counts(a,metadata,report,capsules)
    stage_result=stage_math.stages(stage_records,report,read(roles['raw_exit']),request_sha,sha(roles['run_report']),sha(roles['output_manifest']),a)
    proof.note('independent durable stage/deadline/counter checks')
    epoch=report['epoch_ns'];chosen=report['epoch_chosen_ns'];finished=report['setup_finished_ns']
    if type(epoch) is not int or type(chosen) is not int or epoch-chosen!=200000000 or report['epoch_rebased'] is not False:raise AssertionError('One fixed epoch construction differs')
    if finished is None:
        if len(a['step_index']) or report['first_error'] is None:raise AssertionError('Missing setup finish without a pre-step failure')
        setup_timely=False
    else:
        if type(finished) is not int or finished<chosen:raise AssertionError('Setup finish clock differs')
        setup_timely=finished<epoch
    if len(a['step_index']) and not setup_timely:raise AssertionError('Native steps after missed initial epoch')
    control_result=control.controls(a,metadata,initial[:291],contract,full,hold,epoch)
    reserved_result=control.reserved_control(a,metadata,initial[:291],contract,full,hold,epoch)
    reserved_capture=accounting.uncommitted_capture(a,metadata,report,capsules,initial[:291],contract)
    admission_finish=finished if finished is not None else next(r['record_created_ns'] for r in stage_records if r['stage']=='epoch_armed')
    if len(a['control_control']) and not chosen<=int(a['control_admitted_ns'][0])<=admission_finish:
        raise AssertionError('Initial admission outside chosen epoch setup interval')
    for key in ('event_ledger_overflow','transport_ledger_overflow','outer_ledger_overflow'):
        if key in capsules:metadata[key]=capsules[key]
    outer=[protocol.parsed(value) for value in metadata['outer_cycles']]
    timing_result=physical.timing_arrays(a,outer,epoch,report['session']['foundation']['returned'])
    timing_result['counter_reconstruction']=accounting.saved_timing_counters(a,report,timing_result)
    n=len(a['step_index']);c=len(a['control_control'])
    if metadata['step_command_ids']!=[metadata['control_command_ids'][i//10] for i in range(n)]:raise AssertionError('Saved native command identity differs')
    if len(metadata['step_issues'])!=n or any((issue is None)!=bool(a['step_verified'][i]) or (issue is not None and (type(issue) is not str or not 0<len(issue)<=1024)) for i,issue in enumerate(metadata['step_issues'])):
        raise AssertionError('Saved strict issue mask differs')
    continuity=metadata['hold_continuity']
    if continuity['boundary_control']!=1569 or continuity['present'] is not (c>1569) or continuity['no_reset_in_scaffold'] is not True:
        raise AssertionError('Original terminal hold continuity scope differs')
    if c>1569:
        if not all(continuity[k] is True for k in ('full291_matches_previous_step','history_matches_previous_control','incoming_matches_previous_actual_action')):
            raise AssertionError('Main to hold continuity receipt contradicts checked arrays')
        if continuity['full291_sha256']!=protocol.digest(a['control_integration'][1569].tobytes()) or continuity['history_digest']!=protocol.digest(a['control_flat_history_before'][1569].tobytes()):
            raise AssertionError('Main to hold full291/history identity differs')
    expert_result=physical.compare_expert_prefix(a,full,hold)
    proof.stage='canonical job/mailbox/admission'
    jobs=control.decode(read(output/'issued_jobs.json'));published=read(output/'published_job_activations.json')
    worker=read(output/'worker.json') if (output/'worker.json').exists() else None
    protocol_result=protocol.protocol(a,metadata,jobs,published,worker,run_request,full,hold,sha(roles['expert_main']),sha(roles['expert_hold']),sha(roles['reference']),contract,terminal_interruption=report['first_error'])
    cleanup=report['worker_cleanup'] or {}
    worker_pass=bool(cleanup.get('normal_exit') is True and cleanup.get('pid')==report['worker_pid'] and worker is not None and worker['pid']==report['worker_pid'] and worker['failure'] is None and worker['overflow'] is None and worker['native_steps']==worker['model_calls']==0)
    if worker_pass is not report['worker_pass']:raise AssertionError('Worker raw-success/failure verdict changed')
    physical_pass=physical_result['full_scope'] and physical_result['all_actual_saved_states_strict'] and counter_result['complete_native_coverage']
    timing_pass=timing_result['fixed_epoch_timing_pass'] and setup_timely and stage_result['stage_watchdog_qualified']
    command_pass=control_result['exact_recorded_commands'] and protocol_result['complete_protocol_coverage'] and expert_result['original_expert_prefix_exact']
    qualified=physical_pass and timing_pass and command_pass and worker_pass and mjb_pass and process['observed_processes_absent'] and not process['unobserved_start_side_effects_uncertain'] and report['first_error'] is None and not report['additional_errors'] and report['session']['foundation']['failure'] is None and report['session']['driver_failure'] is None
    if report['component_preliminary_pass'] and not qualified:raise AssertionError('Producer preliminary pass contradicts saved evidence')
    return dict(evidence_integrity_passed=True,physical_pass=physical_pass,timing_pass=timing_pass,command_pass=command_pass,
        component_qualified=bool(qualified),actual_native_steps=counter_result['actual_native_returned'],actual_control_records=control_result['control_count'],
        stage_watchdog=stage_result,stage_sources=stage_source,initial=initial_result,physical=physical_result,counters=counter_result,timing=timing_result,controls=control_result,reserved_control=reserved_result,reserved_capture=reserved_capture,protocol=protocol_result,
        expert_comparison=expert_result,process=process,worker_pass=worker_pass,MJB_serializations=serializations,
        online_policy_qualified=False,realtime_teleoperation_qualified=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=local(args.output);output.mkdir(exist_ok=False);proof=Proof(output);result={};inputs={};initial_sha=sha(local(args.request))
    try:
        request,inputs,roles=validate_request(local(args.request));proof.note('all source and input pins exact')
        result=run_audit(request,roles,proof)
        if sha(local(args.request))!=initial_sha or any(sha(p)!=v for p,v in inputs.items()):raise AssertionError('Input or request changed during saved audit')
        proof.note('final source/input/request pins exact')
    except BaseException as exc:
        result.update(evidence_integrity_passed=False,component_qualified=False,error=repr(exc),traceback=traceback.format_exc(),stage=proof.stage,first_mismatch=proof.first_failure)
    finally:
        proof.stream.close();result.update(request_sha256=initial_sha,comparisons=proof.count,comparisons_sha256=sha(output/'comparisons.jsonl'),
            input_sha256={str(k):v for k,v in inputs.items()},native_steps_executed=0,model_calls=0,optimizer_updates=0,
            runtime={'python':sys.version,'numpy':np.__version__,'platform':sys.platform})
        write(output/'report.json',result)
    return 0 if result['evidence_integrity_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
