"""Gated recorded-command process-clock benchmark. No inference or oracle replay.

Importing this module performs no native call, worker creation or wall-clock tick.
Actual execution requires a separately selected, literal final reviewed request.
"""
import argparse
import dataclasses
import os
import sys
import traceback
from pathlib import Path
import numpy as np
from packet_io import local,sha,read,write,lossless,ready,authorize,pin_check,save_owned,worker_verdict


def error_record(exc):
    return {'type':type(exc).__name__,'detail':str(exc),'traceback':traceback.format_exc()}


def imported_source_paths(pins):
    result={};source=Path(__file__).resolve().parent
    for name in ('packet_io','counted_api','native_loader','model_identity','native_stepper',
                 'clock_core','history','mailbox','shared_mailbox','session','dummy_worker',
                 'recorded_protocol','pending_result','evidence','capture_schema','bfm_observations','oracle_source','stage_watchdog'):
        module=__import__(name);path=Path(module.__file__).resolve()
        if path.parent!=source or pins.get(path)!=sha(path):raise ValueError('wrong imported module: '+name)
        result[name]={'path':str(path),'sha256':sha(path)}
    for name in ('numpy','mujoco'):
        path=Path(sys.modules[name].__file__).resolve()
        if pins.get(path)!=sha(path):raise ValueError('unbound native runtime module: '+name)
        result[name]={'path':str(path),'sha256':sha(path)}
    if any(name.split('.')[0] in ('onnxruntime','torch','mjbatch') for name in sys.modules):
        raise ValueError('inference or private solver runtime imported')
    return result


def require_epoch(chosen,epoch,finished):
    if any(type(v) is not int or v<0 for v in (chosen,epoch,finished)) or not chosen<=finished<epoch or epoch-chosen!=200000000:
        raise RuntimeError('setup/allocation missed the one fixed epoch; no rebase')


def run(request,request_sha,precheck,output,request_path,watchdog,journal):
    # All real imports and side effects occur only after main's concrete gate.
    from stage_watchdog import counters_only
    import multiprocessing as mp
    import time
    from counted_api import CountedAPI,mjb_pair
    from native_stepper import NativeStepper
    from recorded_protocol import canonical_table
    from clock_core import Binding
    from session import DeadlineClock,Session,make_channels
    from dummy_worker import WorkerLifecycle,process_main
    output=Path(output);output.mkdir(exist_ok=False)
    write(output/'pre_input_hashes.json',precheck)
    first_error=None;additional_errors=[];adapter=None;session=None;model=None;api=None
    lifecycle=None;worker_pid=None;cleanup=None;worker_report=None;channels=[]
    epoch_ns=None;epoch_chosen_ns=None;setup_finished_ns=None;exit_identity=None;schema=None
    runtime={};imports={};closed_native=False
    try:
        write(output/'process_start.json',{'pid':os.getpid(),'parent_pid':os.getppid(),'request_sha256':request_sha,
              'proc_stat':Path('/proc/self/stat').read_text()})
        journal.record('native_setup_started',{'watchdog':watchdog.evidence()})
        watchdog.check()
        import mujoco
        from native_loader import load_model
        if sys.platform!='linux' or sys.version_info[:2]!=(3,11) or np.__version__!='1.26.4' or mujoco.__version__!='3.2.3':
            raise ValueError('qualified WSL Python3.11/NumPy1.26.4/MuJoCo3.2.3 required')
        if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')):
            raise ValueError('fixed single-thread runtime environment required')
        pins={local(e['path']).resolve():e['sha256'] for e in request['input_files']}
        imports=imported_source_paths(pins)
        runtime={'python':sys.version,'executable':sys.executable,'numpy':np.__version__,
                 'mujoco':mujoco.__version__,'pid':os.getpid(),'monotonic_clock':time.get_clock_info('monotonic').__dict__}
        roles={k:local(v) for k,v in request['roles'].items()}
        def archive(path):
            with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}
        full=archive(roles['expert_main']);hold=archive(roles['expert_hold']);fixture=archive(roles['canonical_fixture'])
        contract=read(local(request['native_bundle'])/'contract.json')
        limits=np.asarray(contract['joint_limits'],np.float64)
        if int(fixture['state_spec'])!=8191:raise ValueError('canonical fixture spec changed')
        table=canonical_table(full,hold,sha(roles['expert_main']),sha(roles['expert_hold']),limits,fixture['state_vector'])
        if table.sha256!=request['command_table_sha256']:raise ValueError('recorded command schedule changed')
        witness=read(roles['mjb_witness_report']);expected=roles['expected_model_mjb'].read_bytes()
        if witness['passed'] is not True or witness['mjb_sha256']!=request['expected_model_sha256'] or sha(roles['expected_model_mjb'])!=request['expected_model_sha256']:
            raise ValueError('qualified MJB witness differs')
        api=CountedAPI(mujoco,18190,4,output/'mjb')
        model,loaded_contract=load_model(local(request['native_bundle']),'walk003')
        adapter=NativeStepper.__new__(NativeStepper)  # Retain partial constructor on failure.
        NativeStepper.__init__(adapter,model,api.MjData(model),api,loaded_contract,expected,request['expected_model_sha256'])
        adapter.restore_initial(fixture['state_vector'].copy(),full['physics_warning_number'][0].copy(),full['physics_warning_lastinfo'][0].copy())
        binding=Binding(request['run_id'],request['expected_model_sha256'],pins[roles['reference'].resolve()],request['input_epoch'])
        journal.record('native_restored',{'watchdog':watchdog.evidence(),'counters':counters_only(session,adapter,api)})
        watchdog.check()
        context=mp.get_context('spawn');clock=DeadlineClock(time.monotonic_ns,time.sleep)
        jobs,results=make_channels(context,binding);channels=[jobs,results]
        ready_event=context.Event();stop_event=context.Event()
        process=context.Process(target=process_main,args=(jobs,results,binding,table,ready_event,stop_event,str(output/'worker.json')),
                                name='sonic23-recorded-command-worker',daemon=False)
        lifecycle=WorkerLifecycle(process,ready_event,stop_event,clock);lifecycle.start_ready();worker_pid=process.pid
        from multiprocessing import resource_tracker
        write(output/'worker_ready.json',{'worker_pid':worker_pid,'parent_pid':os.getpid(),'lifecycle':lifecycle.records,
            'worker_proc_stat':Path('/proc/'+str(worker_pid)+'/stat').read_text(),
            'resource_tracker_pid':resource_tracker._resource_tracker._pid})
        # The unchanged constructor owns internal ledger allocation. Choose once;
        # allocation must finish before this epoch, otherwise fail without rebasing.
        watchdog.check()
        epoch_chosen_ns=clock.now_ns();epoch_ns=epoch_chosen_ns+request['epoch_lead_ns']
        watchdog.plant(epoch_ns)
        session=Session(clock=clock,stepper=adapter,epoch_ns=epoch_ns,binding=binding,table=table,
                        initial_previous_raw=full['previous_action'][0].tobytes(),limits=limits,jobs=jobs,results=results,
                        main_controls=1569,event_capacity=80000,transport_capacity=80000,max_debt_steps=100,max_elapsed_ns=60000000000)
        allocation_finished_ns=clock.now_ns()
        require_epoch(epoch_chosen_ns,epoch_ns,allocation_finished_ns)
        journal.record('epoch_armed',{'epoch_chosen_ns':epoch_chosen_ns,'epoch_ns':epoch_ns,
            'allocation_finished_ns':allocation_finished_ns,'watchdog':watchdog.evidence(),
            'counters':counters_only(session,adapter,api),'worker_pid':worker_pid})
        setup_finished_ns=clock.now_ns()
        require_epoch(epoch_chosen_ns,epoch_ns,setup_finished_ns)
        while session.foundation.returned<18190:
            if not session.tick_once():break
        # No post-failure command, reset, additional snapshot, inference or retry.
    except BaseException as exc:first_error=error_record(exc)
    finally:
        # This first small record precedes cleanup, MJB exit, trace export and hashes.
        watchdog.preserve()
        journal.record('plant_or_setup_exit',{'first_error':first_error,'epoch_chosen_ns':epoch_chosen_ns,
            'epoch_ns':epoch_ns,'setup_finished_ns':setup_finished_ns,'watchdog':watchdog.evidence(),
            'counters':counters_only(session,adapter,api)})
        watchdog.check()
        if lifecycle is not None and lifecycle.start_attempted and not lifecycle.closed:
            try:cleanup=lifecycle.stop_join()
            except BaseException as exc:additional_errors.append({'phase':'worker_cleanup',**error_record(exc)})
        watchdog.check()
        if (output/'worker.json').exists():
            try:worker_report=read(output/'worker.json')
            except BaseException as exc:additional_errors.append({'phase':'worker_report',**error_record(exc)})
        # Required model exit even if setup, native physics, timing or worker fails.
        watchdog.check()
        if model is not None and api is not None:
            try:
                if session is not None:exit_identity=session.close_native();closed_native=True
                elif adapter is not None and hasattr(adapter,'identity'):
                    exit_identity=adapter.verify_model_exit();closed_native=True
                else:
                    _,exit_identity=mjb_pair(model,api,output,'constructor_failure_exit',expected);closed_native=True
            except BaseException as exc:additional_errors.append({'phase':'model_exit',**error_record(exc)})
        watchdog.check()
        for channel in channels:
            try:channel.close()
            except BaseException as exc:additional_errors.append({'phase':'endpoint_close',**error_record(exc)})
        watchdog.check()
        if session is not None:
            try:
                schema=save_owned(output,session)
                write(output/'issued_jobs.json',{str(k):lossless(v.to_bytes()) for k,v in session.foundation.issued.items()})
                write(output/'published_job_activations.json',sorted(session.foundation.published_jobs))
            except BaseException as exc:additional_errors.append({'phase':'owned_evidence_writer',**error_record(exc)})
        else:
            # Preserve available already-owned setup evidence without new native reads.
            setup={}
            if adapter is not None:
                for name in ('failure','initial_capture','last_capture','last_capture_return'):
                    value=getattr(adapter,name,None)
                    if value is not None:setup[name]=lossless(value)
            try:write(output/'setup_evidence.json',setup)
            except BaseException as exc:additional_errors.append({'phase':'setup_evidence_writer',**error_record(exc)})
        watchdog.check()
        try:postcheck=pin_check(request['input_files'])
        except BaseException as exc:
            postcheck={'all_exact':False,'files':{},'error':error_record(exc)}
        watchdog.check()
        write(output/'post_input_hashes.json',postcheck)
        if not postcheck['all_exact']:additional_errors.append({'phase':'postrun_pins','detail':'input changed'})
        try:request_unchanged=sha(request_path)==request_sha
        except BaseException:request_unchanged=False
        if not request_unchanged:additional_errors.append({'phase':'postrun_request','detail':'request changed'})
        try:summary=None if session is None else session.summary()
        except BaseException as exc:
            summary=None;additional_errors.append({'phase':'summary',**error_record(exc)})
        worker_pass=worker_verdict(cleanup,worker_report,worker_pid)
        api_counts=None if api is None else api.counters()
        if api_counts is not None:
            serialization_pass=(api_counts['serialization_attempted']==api_counts['serialization_returned']==4 and
                len(api_counts['serialization_records'])==4 and all(v.get('sha256')==request['expected_model_sha256']
                and v.get('native_returned') is True and not v.get('capture_error') for v in api_counts['serialization_records']))
        else:serialization_pass=False
        preliminary=bool(first_error is None and not additional_errors and worker_pass and serialization_pass and
            summary is not None and summary['component_preliminary_pass'] and schema is not None)
        result={'kind':'recorded_command_process_clock_component','request_sha256':request_sha,
                'requested_controls':1819,'main_controls':1569,'hold_controls':250,'requested_native_steps':18190,
                'component_preliminary_pass':preliminary,'component_qualified':False,'root_saved_audit_pending':True,
                'online_policy_qualified':False,'realtime_controller_qualified':False,'model_calls':0,'optimizer_updates':0,
                'other_oracle_native_steps':0,'first_error':first_error,'additional_errors':additional_errors,
                'session':summary,'api_counters':api_counts,'worker_pass':worker_pass,'worker_cleanup':cleanup,
                'worker_pid':worker_pid,'worker_failure':None if worker_report is None else worker_report.get('failure'),
                'worker_report_present':worker_report is not None,'worker_lifecycle':[] if lifecycle is None else lifecycle.records,
                'epoch_chosen_ns':epoch_chosen_ns,'epoch_ns':epoch_ns,'setup_finished_ns':setup_finished_ns,
                'epoch_rebased':False,'exit_model_identity':exit_identity,'model_exit_attempted':model is not None,
                'all_four_MJB_exact':serialization_pass,'schema':schema,'runtime':runtime,'imported_sources':imports,
                'pre_and_post_input_hashes_exact':precheck['all_exact'] and postcheck['all_exact'],
                'stage_watchdog':watchdog.evidence()}
        watchdog.check()
        write(output/'report.json',result)
        watchdog.check()
        outputs={str(p.relative_to(output)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in output.rglob('*') if p.is_file()}
        write(output/'output_manifest.json',{'request_sha256':request_sha,'files':outputs})
        watchdog.check()
        journal.record('preservation_complete',{'watchdog':watchdog.evidence(),
            'counters':counters_only(session,adapter,api),'report_sha256':sha(output/'report.json'),
            'output_manifest_sha256':sha(output/'output_manifest.json')})
        watchdog.finish()
    return 0 if result['component_preliminary_pass'] else 2


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--clearance',required=True)
    args=parser.parse_args()
    # Literal permission/source review first; expensive pin and native setup are
    # covered by the separate setup deadline, not the fixed plant deadline.
    request,request_sha=authorize(args.request,args.clearance)
    import time
    from stage_watchdog import PosixWatchdog,StageJournal
    journal=StageJournal(local(request['stage_directory']),time.monotonic_ns,request_sha)
    with PosixWatchdog(time.monotonic_ns) as watchdog:
        watchdog.setup()
        journal.record('setup_started',{'watchdog':watchdog.evidence()})
        try:
            checked,checked_sha,checks=ready(args.request,args.clearance)
            if checked_sha!=request_sha:raise ValueError('request changed during setup')
            return run(checked,checked_sha,checks,local(request['output_directory']),local(args.request),watchdog,journal)
        except BaseException as exc:
            # A missing final run report is incomplete evidence, never zero steps.
            if watchdog.phase in ('setup','plant'):watchdog.preserve()
            journal.record('incomplete_process_exit',{'error':error_record(exc),'watchdog':watchdog.evidence(),
                'native_counts_not_inferred':True,'automatic_retry':False})
            return 2



if __name__=='__main__':raise SystemExit(main())
