"""Saved native-adapter equivalence audit. No model, native API or optimizer imports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from saved_math import exact,compare_case,compare_fault

def local(value):
    value=str(value).replace('\\','/')
    if sys.platform!='win32' and len(value)>2 and value[1]==':':value='/mnt/'+value[0].lower()+value[2:]
    return Path(value)
def sha(path):
    h=hashlib.sha256()
    with local(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(local(path).read_text(encoding='utf-8-sig'))
def archive(path):
    with np.load(local(path),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def contains(value,digest):
    if isinstance(value,dict):return any(contains(v,digest) for v in value.values())
    if isinstance(value,list):return any(contains(v,digest) for v in value)
    return value==digest

def identity(record,mjb_sha):
    assert record['expected_sha256']==mjb_sha and record['entry_verified'] is True and record['exit_verified'] is True
    assert record['serialization_attempted']==record['serialization_returned']==4
    assert record['process_global_identity_proven'] is False and record['portable_cross_platform_identity'] is False
    assert len(record['checks'])==2
    for index,(check,phase) in enumerate(zip(record['checks'],('entry','exit'))):
        assert check['phase']==phase and check['passed'] is True and check['actual_sha256']==mjb_sha
        for field in ('serialization_attempted','serialization_returned'):
            assert check[field+'_before']==2*index and check[field+'_after']==2*index+2

def api_counts(report,steps,serializations):
    assert report['step_budget']==steps and report['serialization_budget']==serializations
    assert report['step_attempted']==report['step_returned']==steps
    assert report['serialization_attempted']==report['serialization_returned']==serializations
    assert report['denied_step_calls']==report['denied_serialization_calls']==0

def serialization_records(records,buffers,expected):
    assert len(records)==len(buffers)
    digest=hashlib.sha256(expected).hexdigest()
    for index,(record,buffer) in enumerate(zip(records,buffers)):
        assert record['index']==index and record['native_returned'] is True
        assert 'error' not in record and 'capture_error' not in record
        assert record['file']==f'{index:02d}.mjb' and record['dtype']=='|u1' and record['shape']==[len(expected)]
        assert record['bytes']==len(expected) and record['sha256']==digest and buffer==expected

def run(request_path,output):
    request_path=local(request_path);output=local(output);output.mkdir(exist_ok=False)
    tracked={};checks=[];context={};result={'passed':False}
    def bind(path,expected=None):
        path=local(path).resolve();digest=sha(path)
        if expected is not None:assert digest==expected,str(path)+' subject changed'
        if str(path) in tracked:assert tracked[str(path)]==digest,str(path)+' changed during audit'
        tracked[str(path)]=digest
        return path
    def role(value):return bind(value['path'],value['sha256'])
    def passed(name):checks.append({'name':name,'passed':True})
    try:
        request=read(bind(request_path));request_sha=sha(request_path)
        assert request['kind']=='saved_native_stepper_equivalence_audit' and request['completed_native_run_required'] is True
        for path in [Path(__file__),Path(__file__).with_name('saved_math.py')]:bind(path)
        for subject in request['source_files']:role(subject)
        assert all(any(local(v['path']).resolve()==p.resolve() and v['sha256']==sha(p) for v in request['source_files']) for p in [Path(__file__),Path(__file__).with_name('saved_math.py')])
        assert request['budgets']=={'model_calls':0,'native_steps':0,'optimizer_updates':0}
        phases={}
        for phase,steps,serializations in [('witness',0,2),('replay',21348,8)]:
            context={'phase':phase,'stage':'completed_lineage'}
            spec=request['stages'][phase]
            producer_request_path=role(spec['request']);producer_request=read(producer_request_path)
            report_path=role(spec['report']);report=read(report_path)
            clearance_path=role(spec['clearance']);clearance=read(clearance_path)
            assert producer_request['stage']==phase and producer_request['source_preparation_only'] is False
            assert producer_request['native_step_budget']==steps and producer_request['serialization_budget']==serializations
            assert producer_request['model_inference_calls']==producer_request['optimizer_updates']==0
            assert producer_request['plant_foundation_connected'] is False
            assert report['passed'] is True and report['error'] is None and report['request_sha256']==sha(producer_request_path)
            assert report['model_inference_calls']==report['optimizer_updates']==report['other_oracle_native_steps']==0
            assert report['plant_foundation_connected'] is False and report['real_wallclock_experiment'] is False and report['hardware_authorized'] is False
            api_counts(report['api_counters'],steps,serializations)
            assert clearance['root_selected_single_run'] is True and clearance['request_sha256']==sha(producer_request_path)
            source_review=read(role(clearance['review']))
            assert source_review[clearance['review']['pass_field']] is True
            assert local(source_review['request_subject']['path']).resolve()==producer_request_path
            assert source_review['request_subject']['sha256']==sha(producer_request_path)
            producer_pins={}
            for item in producer_request['input_files']:
                path=role(item)
                if str(path) in producer_pins:assert producer_pins[str(path)]==item['sha256']
                producer_pins[str(path)]=item['sha256']
            owner_path=role(spec['owner']);owner=read(owner_path)
            assert owner['passed'] is True and owner['pre_and_post_input_hashes_exact'] is True
            assert owner['request_sha256']==sha(producer_request_path) and owner['report_sha256']==sha(report_path)
            assert owner['clearance_sha256']==sha(clearance_path)
            exit_path=role(spec['process_exit']);exit_receipt=read(exit_path)
            assert exit_receipt['exit_code']==0 and exit_receipt.get('error') is None
            assert exit_receipt['raw_python_exit_code']==0 and exit_receipt['diagnostic_exit_code']==0
            assert exit_receipt['all_postrun_hashes_exact'] is True
            assert owner['process_exit_sha256']==sha(exit_path)
            launch_path=role(spec['launch_receipt']);assert owner['launch_receipt_sha256']==sha(launch_path)
            absence_path=role(spec['process_absence']);absence=read(absence_path)
            assert owner['process_absence_sha256']==sha(absence_path)
            assert absence['wrapper_absent'] is True and absence['child_absent'] is True
            for key in ('wrapper_pid','child_pid','wrapper_absent','child_absent'):assert owner['process_absence'][key]==absence[key]
            assert absence['wrapper_pid']==exit_receipt['wrapper_pid'] and absence['child_pid']==exit_receipt['child_pid']
            owner_pins={str(local(path).resolve()):digest for path,digest in owner['input_hashes'].items()}
            for path,digest in owner_pins.items():bind(path,digest)
            for path,digest in producer_pins.items():assert owner_pins[path]==digest
            for phase_name in ('pre_hashes','post_hashes'):
                ledger_path=role(spec[phase_name]);ledger=read(ledger_path)
                assert ledger['all_exact'] is True
                records={str(local(path).resolve()):value for path,value in ledger['files'].items()}
                assert set(records)==set(owner_pins)
                for path,digest in owner_pins.items():
                    assert records[path]['expected']==records[path]['actual']==digest and records[path]['matched'] is True
            for path,digest in owner['output_hashes'].items():bind(path,digest)
            for item in spec['completion_evidence']:role(item)
            # The concrete request lists actual entry/exit/runtime identities; no recursive training traversal.
            phases[phase]=(producer_request,report,report_path.parent,producer_pins)
            passed(phase+'_completed_request_clearance_owner_exit_and_all_input_pins')
        witness,witness_report,witness_dir,_=phases['witness'];replay,replay_report,replay_dir,replay_pins=phases['replay']
        expected_mjb=bind(witness_dir/'expected_model.mjb').read_bytes();mjb_sha=hashlib.sha256(expected_mjb).hexdigest()
        assert witness_report['mjb_sha256']==mjb_sha and witness_report['mjb_bytes']==len(expected_mjb)
        for index in (0,1):assert bind(witness_dir/f'independent_witness_{index}.mjb').read_bytes()==expected_mjb
        serialization=read(bind(witness_dir/'independent_witness_serialization.json'))
        assert serialization==witness_report['serialization'] and serialization['passed'] is True
        assert serialization['returned_files']==['independent_witness_0.mjb','independent_witness_1.mjb'] and serialization['current_returned'] is True
        assert serialization['mjb_sha256']==mjb_sha and serialization['bytes']==len(expected_mjb)
        api_counts(serialization['api_counters'],0,2)
        for report,directory,count in [(witness_report,witness_dir,2),(replay_report,replay_dir,8)]:
            records=report['api_counters']['serialization_records'];assert len(records)==count
            buffers=[bind(directory/'serialization_buffers'/f'{index:02d}.mjb').read_bytes() for index in range(count)]
            serialization_records(records,buffers,expected_mjb)
        assert local(replay['expected_model_mjb']).resolve()==(witness_dir/'expected_model.mjb').resolve()
        assert local(replay['mjb_witness_report']).resolve()==(witness_dir/'report.json').resolve()
        for p in [witness_dir/'expected_model.mjb',witness_dir/'report.json']:
            assert replay_pins[str(p.resolve())]==sha(p)
        passed('independent_two_buffer_witness_MJB_bytes_and_replay_subjects')
        contract=read(role(request['native_contract']));contract={k:np.asarray(v,np.float64) for k,v in contract.items() if k in ('kp','kd','native_effort','native_velocity','joint_limits')}
        assert set(contract)=={'kp','kd','native_effort','native_velocity','joint_limits'}
        assert local(request['native_contract']['path']).resolve()==(local(replay['native_bundle'])/'contract.json').resolve()
        assert local(witness['native_bundle']).resolve()==local(replay['native_bundle']).resolve()
        contract_path=local(request['native_contract']['path']).resolve()
        assert replay_pins[str(contract_path)]==sha(contract_path)
        fixture_path=role(request['canonical_fixture']);fixture=archive(fixture_path)
        assert local(replay['canonical_fixture']).resolve()==fixture_path
        assert replay_pins[str(fixture_path)]==sha(fixture_path)
        assert sha(fixture_path)=='4f99f1bd0d9559ec23bd0a73afe5e49a711a90c3dcecafd1a121bae89b0dbde3'
        traces={}
        expected_trace_hashes={'expert_main':'721a44a88ba3c8d1c9a42c23ab3ec1d0b39b529c78abaa54df19f07ebf5a886a',
            'expert_hold':'e171f7e3cc6a1b08484f4f6d328bc43ebc4ff6a39555416157c4d5191d713780',
            'direct_failure':'449a31d17aa7df6b23c20d6b0394237fc36b97d779f46544e6375975e0cdf427'}
        for name in ['expert_main','expert_hold','direct_failure']:
            path=role(request['qualified_traces'][name])
            assert sha(path)==expected_trace_hashes[name]
            assert local(replay['traces'][name]).resolve()==path and replay_pins[str(path)]==sha(path)
            traces[name]=archive(path)
        for name,subject in request['qualification_reports'].items():
            qualification=read(role(subject))
            assert qualification['recorded_trace_reproduced_through_last_sample'] is True
            assert qualification['physics_steps']=={'expert_main':15690,'expert_hold':2500,'direct_failure':3158}[name]
            assert qualification['feasible'] is (name!='direct_failure')
            assert contains(qualification,expected_trace_hashes[name])
            assert len(qualification['original_trace_comparison'])==7 and all(v is True for v in qualification['original_trace_comparison'].values())
        assert set(request['qualification_reports'])==set(traces)
        assert traces['expert_main']['target'].shape==(1569,23) and traces['expert_hold']['target'].shape==(250,23) and traces['direct_failure']['target'].shape==(316,23)
        for name,start,count in [('expert_main',0,1569),('expert_hold',1569,250),('direct_failure',0,316)]:
            exact(traces[name]['global_control'],np.arange(start,start+count,dtype=np.int64),'fixed original control range '+name)
            substeps=np.full(count,10,np.int64)
            if name=='direct_failure':substeps[-1]=8
            exact(traces[name]['physics_substeps'],substeps,'fixed original substeps '+name)
        assert int(fixture['state_spec'])==8191
        for name in ('expert_main','direct_failure'):exact(fixture['state_vector'],traces[name]['initial_integration'],'canonical fixture '+name)
        case_results={}
        for case,n,verified,names in [('expert_continuous',18190,18190,['expert_main','expert_hold']),('direct_failure',3158,3157,['direct_failure'])]:
            context={'phase':'replay','case':case,'stage':'saved_samples'}
            directory=replay_dir/case;case_report=read(bind(directory/'report.json'));captured=archive(bind(directory/'captured_trace.npz'))
            assert case_report['equivalence_passed'] is True and case_report['error'] is None and case_report['first_mismatch'] is None
            assert case_report['model_inference_calls']==case_report['other_oracle_native_steps']==0
            assert case_report['no_reset_between_segments'] is True and case_report['wallclock_or_balance_qualification'] is False
            counters=case_report['adapter_counters']
            for key in ('attempted','returned','capture_attempts','captured','verification_attempts'):assert counters[key]==n
            assert counters['verified']==verified and counters['initialized'] is True and counters['closed'] is True and counters['mutation_uncertain'] is False
            identity(counters['model_identity'],mjb_sha)
            assert case_report['exit_model_identity']==counters['model_identity']['checks'][-1]
            case_results[case],decoded=compare_case(captured,[(name,traces[name]) for name in names],contract)
            assert case_results[case]['samples']==n
            exact(np.asarray(counters['expected_time'],np.float64),captured['simulation_time'][-1],case+' repeated expected endpoint')
            exact(np.asarray(counters['initial_time'],np.float64),fixture['state_vector'][0],case+' initial endpoint')
            expected_failures=[] if case=='expert_continuous' else [{'row':3157,'control':315,'substep':8,'reasons':['native_joint_bound']}]
            assert case_results[case]['strict_failures']==expected_failures
            assert len(case_report['segment_results'])==len(names)
            for item,name in zip(case_report['segment_results'],names):
                assert item['segment']==name and item['all_saved_samples_byte_exact'] is True
                assert item['reproduced_steps']==int(traces[name]['physics_substeps'].sum())
                issue=None if name!='direct_failure' else [315,8,'native_joint_bound']
                assert item['expected_issue']==item['actual_issue']==issue and item['requested_segment_complete']==(issue is None)
            if case=='expert_continuous':assert case_report['adapter_fault'] is None
            else:
                fault_path=bind(directory/'adapter_fault_evidence.bin',case_report['adapter_fault']['evidence_sha256'])
                capture_path=bind(directory/'adapter_capture_return_evidence.bin',case_report['adapter_fault']['capture_return_evidence_sha256'])
                case_results[case]['fault']=compare_fault(fault_path.read_bytes(),capture_path.read_bytes(),case_report['adapter_fault'],traces['direct_failure'],captured)
            before=case_report['initial_api_counters'];after=case_report['final_api_counters']
            assert before['step_attempted']==before['step_returned']==(0 if case=='expert_continuous' else 18190)
            assert before['serialization_attempted']==before['serialization_returned']==(0 if case=='expert_continuous' else 4)
            assert after['step_attempted']==after['step_returned']==(18190 if case=='expert_continuous' else 21348)
            assert after['serialization_attempted']==after['serialization_returned']==(4 if case=='expert_continuous' else 8)
            assert after['denied_step_calls']==after['denied_serialization_calls']==0
            assert before['serialization_records']==replay_report['api_counters']['serialization_records'][:(0 if case=='expert_continuous' else 4)]
            assert after['serialization_records']==replay_report['api_counters']['serialization_records'][:(4 if case=='expert_continuous' else 8)]
            assert replay_report['expert' if case=='expert_continuous' else 'direct']==case_report
            passed(case+'_all_samples_boundaries_identity_counts_and_expected_fault')
        assert replay_report['independent_witness_serializations']==2 and replay_report['total_serializations_including_witness']==10
        assert replay_report['original_expert_physics_pass'] is True and replay_report['original_direct_physics_pass'] is False
        assert replay_report['exact_expected_failure_reproduction'] is True and replay_report['no_replayed_hold_after_failed_direct'] is True
        for path,digest in tracked.items():assert sha(path)==digest,path+' final rehash failed'
        assert sha(request_path)==request_sha
        result=dict(passed=True,kind='independent_saved_native_adapter_equivalence',request_sha256=request_sha,cases=case_results,
            reproduced_native_samples=21348,expert_physics_pass=True,direct_physics_pass=False,
            adapter_equivalence_pass=True,witness_serializations=2,replay_serializations=8,
            all_boundaries_full291_exact=True,all_source_input_output_hashes_unchanged=True,
            checks=len(checks),input_sha256=tracked,model_calls=0,native_steps=0,optimizer_updates=0,
            limitations=['No replay or model execution by this saved auditor.',
                'All ten actual serialization buffers are compared byte-for-byte; this does not prove process-global native pointer identity or cross-platform portability.',
                'Expected direct physical failure is preserved. Adapter equivalence is not balance, asynchronous clock, real-time or hardware qualification.'])
    except BaseException as exc:
        result=dict(passed=False,error=repr(exc),traceback=traceback.format_exc(),context=context,checks=len(checks),input_sha256=tracked,
                    model_calls=0,native_steps=0,optimizer_updates=0)
        raise
    finally:
        (output/'checks.json').write_text(json.dumps(checks,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'report_sha256':sha(output/'report.json'),'steps_checked':21348}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.request,args.output)
