"""Future reviewed adapter equivalence run; no inference or other oracle replay."""
import argparse,hashlib,json,sys,traceback
from pathlib import Path
import numpy as np
from counted_api import CountedAPI,mjb_pair
from replay_core import Segment,Evidence,archive,replay_segment,validate_segment,exact

BASE=Path(__file__).resolve().parent.parent

def local(p):
    text=str(p).replace('\\','/')
    if sys.platform!='win32' and len(text)>2 and text[1]==':':text='/mnt/'+text[0].lower()+text[2:]
    return Path(text)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def has_hash(v,d):
    if isinstance(v,dict):return any(has_hash(x,d) for x in v.values())
    if isinstance(v,list):return any(has_hash(x,d) for x in v)
    return v==d
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def require_review_subject(review,request_path,request_sha):
    subject=review['request_subject']
    assert local(subject['path']).resolve()==Path(request_path).resolve(),'review request path differs'
    assert subject['sha256']==request_sha,'review request SHA differs'

def require_pinned_roles(r,pins):
    def required(value):
        p=local(value)
        assert p.is_absolute() and p.resolve() in pins,'consumed role path is not pinned: '+str(p)
        assert p.is_file() and sha(p)==pins[p.resolve()],'consumed role changed: '+str(p)
        return p
    bundle=local(r['native_bundle']);assert bundle.is_absolute() and bundle.is_dir()
    manifest=read(required(bundle/'manifest.json'))
    for name in ('native_prepared.xml','prepared_model_arrays.npz','contract.json','walk003/native_original.npz'):required(bundle/name)
    assert sha(bundle/'native_prepared.xml')==manifest['portable_xml_sha256']
    assert sha(bundle/'prepared_model_arrays.npz')==manifest['prepared_arrays_sha256']
    assert sha(bundle/'walk003/native_original.npz')==manifest['cases']['walk003']['native_original.npz']
    for name,digest in manifest['meshes'].items():
        p=(bundle/'meshes'/name).resolve()
        assert p.is_relative_to((bundle/'meshes').resolve()),'mesh escapes native bundle'
        assert sha(required(p))==digest
    if r['stage']=='replay':
        required(r['canonical_fixture'])
        assert set(r['traces'])=={'expert_main','expert_hold','direct_failure'}
        for p in r['traces'].values():required(p)
        required(r['mjb_witness_report']);required(r['expected_model_mjb'])

def ready(request_path,clearance_path):
    request_path=local(request_path);assert request_path.is_absolute()
    r=read(request_path);c=read(local(clearance_path))
    assert c['root_selected_single_run'] is True and c['request_sha256']==sha(request_path)
    review_path=local(c['review']['path']);assert sha(review_path)==c['review']['sha256']
    review=read(review_path);assert review[c['review']['pass_field']] is True
    require_review_subject(review,request_path,sha(request_path))
    assert r['stage'] in ('witness','replay') and r['source_preparation_only'] is False
    expected=(0,2) if r['stage']=='witness' else (21348,8)
    assert (r['native_step_budget'],r['serialization_budget'])==expected
    assert r['model_inference_calls']==r['optimizer_updates']==0 and r['plant_foundation_connected'] is False
    pins={}
    for entry in r['input_files']:
        p=local(entry['path']);assert p.is_absolute() and p.is_file() and sha(p)==entry['sha256'],str(p)
        pins[p.resolve()]=entry['sha256']
    for p in Path(__file__).parent.rglob('*.py'):assert p.resolve() in pins and sha(p)==pins[p.resolve()]
    require_pinned_roles(r,pins)
    return r,sha(request_path)

def run_case(model,contract,api,expected,segments,fixture,output):
    from native_stepper import NativeStepper
    proof=Evidence(output);adapter=None;initialized=False;error=None;exit_record=None
    initial_api=api.counters();result={};case_passed=False
    # Allocate separately so partial constructor counters/evidence remain accessible.
    adapter=NativeStepper.__new__(NativeStepper)
    try:
        NativeStepper.__init__(adapter,model,api.MjData(model),api,contract,expected,hashlib.sha256(expected).hexdigest())
        initialized=True
        first=segments[0].arrays
        if not exact(first['initial_integration'],fixture['state_vector']):raise AssertionError('canonical complete291 fixture mismatch')
        if int(fixture['state_spec'])!=8191:raise AssertionError('canonical fixture schema')
        warning_key='physics_warning_number' if 'physics_warning_number' in first else 'physics_warning_counts'
        adapter.restore_initial(fixture['state_vector'].copy(),first[warning_key][0].copy(),first['physics_warning_lastinfo'][0].copy())
        for segment in segments:replay_segment(adapter,segment,proof)
        expected_steps=sum(s.recorded_steps for s in segments);expected_failed=sum(s.expected_issue is not None for s in segments)
        counters=adapter.counters()
        assert counters['attempted']==counters['returned']==counters['capture_attempts']==counters['captured']==counters['verification_attempts']==expected_steps
        assert counters['verified']==expected_steps-expected_failed
        assert len(proof.captures)==expected_steps
        if expected_failed:
            assert adapter.failure is not None and adapter.failure.reason=='STRICT_NATIVE_VIOLATION'
            assert adapter.failure.detail=='native_joint_bound' and adapter.failure.returned==3158
        else:assert adapter.failure is None
        case_passed=True
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),current=dict(proof.current))
    finally:
        # Full model exit identity is required even when comparison or stepping fails.
        # A constructor may fail before self.identity is assigned; preserve it and
        # use exactly one independent two-buffer exit comparison in that case.
        try:
            if initialized or hasattr(adapter,'identity'):
                exit_record=adapter.verify_model_exit()
            else:
                _,exit_record=mjb_pair(model,api,proof.output,'constructor_failure_exit',expected)
        except BaseException as exc:
            exit_record=dict(passed=False,error=repr(exc));case_passed=False
            if error is None:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        try:
            proof.save('captured_trace.npz' if case_passed else 'partial_captured_trace.npz')
            fault=proof.fault(adapter) if hasattr(adapter,'failure') else None
            counters=adapter.counters() if hasattr(adapter,'identity') else dict(partial_constructor=True)
            result=dict(equivalence_passed=case_passed,error=error,first_mismatch=proof.first_mismatch,
                comparisons=proof.comparisons,segment_results=proof.segment_results,adapter_fault=fault,
                adapter_counters=counters,initial_api_counters=initial_api,final_api_counters=api.counters(),
                exit_model_identity=exit_record,no_reset_between_segments=True,
                model_inference_calls=0,other_oracle_native_steps=0,wallclock_or_balance_qualification=False)
            write(proof.output/'report.json',result)
        except BaseException as exc:
            write(proof.output/'preservation_failure.json',dict(error=repr(exc),original_error=error,api_counters=api.counters()))
            raise
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--clearance',type=Path,required=True);a=p.parse_args()
    r,request_sha=ready(a.request,a.clearance)
    output=BASE/r['stage'];output.mkdir(exist_ok=False)
    api=None;error=None;result={};raw_buffers=None
    try:
        assert sys.platform!='win32' and np.__version__=='1.26.4'
        import mujoco
        from native_loader import load_model
        assert mujoco.__version__=='3.2.3'
        api=CountedAPI(mujoco,r['native_step_budget'],r['serialization_budget'],output/'serialization_buffers')
        bundle=local(r['native_bundle'])
        if r['stage']=='witness':
            model,contract=load_model(bundle,'walk003')
            raw_buffers,proof=mjb_pair(model,api,output,'independent_witness')
            (output/'expected_model.mjb').write_bytes(raw_buffers)
            assert api.serialization_attempted==api.serialization_returned==2 and api.step_attempted==api.step_returned==0
            result=dict(passed=True,kind='independent_zero_step_complete_MJB_witness',mjb_sha256=sha(output/'expected_model.mjb'),
                mjb_bytes=len(raw_buffers),serialization=proof)
        else:
            witness=read(local(r['mjb_witness_report']));expected=local(r['expected_model_mjb']).read_bytes()
            assert witness['passed'] is True and witness['mjb_sha256']==hashlib.sha256(expected).hexdigest()
            assert witness['api_counters']['step_attempted']==witness['api_counters']['step_returned']==0
            assert witness['api_counters']['serialization_attempted']==witness['api_counters']['serialization_returned']==2
            fixture=archive(local(r['canonical_fixture']))
            z={name:archive(local(path)) for name,path in r['traces'].items()}
            expert=[Segment('expert_main',z['expert_main'],0,1569,1569,15690),Segment('expert_hold',z['expert_hold'],1569,250,250,2500)]
            failure=[Segment('direct_failure',z['direct_failure'],0,316,1569,3158,(315,8,'native_joint_bound'))]
            for segment in expert+failure:validate_segment(segment)
            assert exact(expert[0].arrays['final_integration'],expert[1].arrays['initial_integration'])
            for segment in (expert[0],failure[0]):assert exact(segment.arrays['initial_integration'],fixture['state_vector'])
            model,c=load_model(bundle,'walk003');full=run_case(model,c,api,expected,expert,fixture,output/'expert_continuous')
            if not full['equivalence_passed']:
                write(output/'direct_failure_unrun.json',dict(requested_native_steps=3158,actual_native_steps=0,
                    reason='expert adapter equivalence failed; no retry or direct case execution'))
                raise RuntimeError('expert adapter equivalence failed; direct case unrun')
            model,c=load_model(bundle,'walk003');failed=run_case(model,c,api,expected,failure,fixture,output/'direct_failure')
            assert failed['equivalence_passed'],'saved direct failure not reproduced by adapter'
            assert api.step_attempted==api.step_returned==21348 and api.serialization_attempted==api.serialization_returned==8
            result=dict(passed=True,kind='recorded_native_adapter_equivalence',expert=full,direct=failed,
                independent_witness_serializations=2,total_serializations_including_witness=10,
                original_expert_physics_pass=True,original_direct_physics_pass=False,
                exact_expected_failure_reproduction=True,no_replayed_hold_after_failed_direct=True)
        final,final_sha=ready(a.request,a.clearance);assert final_sha==request_sha
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());result['passed']=False
    finally:
        result.update(request_sha256=request_sha,error=error,api_counters=None if api is None else api.counters(),
            model_inference_calls=0,optimizer_updates=0,other_oracle_native_steps=0,
            plant_foundation_connected=False,real_wallclock_experiment=False,hardware_authorized=False)
        write(output/'report.json',result)
    if not result['passed']:raise SystemExit(2)

if __name__=='__main__':main()
