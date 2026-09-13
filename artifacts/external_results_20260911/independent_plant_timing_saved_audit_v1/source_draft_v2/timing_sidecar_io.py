"""Read only bound sidecars; independent decoding and failure-aware prefix math."""
import hashlib,json
import numpy as np
from clock_control_math import decode
from qualified_binary_math import typed_json
import clock_saved_math as physical
import timing_saved_math

NAMES={'timing_spans.bin','timing_gc.bin','timing_metadata.json','timing_owned_partial.json'}

def read_sidecars(output,report,stages,arrays,pinned):
    declaration=report.get('timing_instrumentation')
    if declaration is None:
        if report['component_preliminary_pass'] or not report['additional_errors']:raise AssertionError('Missing sidecars without failed preservation')
        return dict(sidecar_evidence_integrity_passed=True,instrumentation_complete=False,sidecar_available=False,
            counter_handoff={'partial_counter_proof':False}),{}
    if declaration['enabled'] is not True or type(declaration['new_native_reads']) is not int or declaration['new_native_reads']!=0 or type(declaration['complete']) is not bool:
        raise AssertionError('Sidecar execution/qualification declaration differs')
    files=declaration['files'];errors=declaration['errors']
    if not set(files)<=NAMES or type(errors) is not list:raise AssertionError('Unexpected timing output schema')
    data={}
    for name,entry in files.items():
        path=(output/name).resolve();raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        if sha!=entry['sha256'] or len(raw)!=entry['bytes'] or pinned.get(path)!=sha:raise AssertionError('Unbound or changed timing sidecar')
        data[name]=raw
    if set(files)!=NAMES:
        if declaration['complete'] or not errors:raise AssertionError('Missing sidecar without writer failure')
        return dict(sidecar_evidence_integrity_passed=True,instrumentation_complete=False,sidecar_available=False,
            retained_files=sorted(files),writer_errors=errors,counter_handoff={'partial_counter_proof':False}),{}
    metadata=json.loads(data['timing_metadata.json']);owned=decode(json.loads(data['timing_owned_partial.json']))
    result=timing_saved_math.audit(metadata,data['timing_spans.bin'],data['timing_gc.bin'],report,stages,arrays,owned)
    result['sidecar_available']=True;result['writer_errors']=errors
    return result,owned


def reserved_capture(a,metadata,report,capsules,initial,contract,owned):
    """Check actual owned last capture without inventing a verification return."""
    n=len(a['step_index']);core=report['session']['foundation'];native=report['session']['stepper']
    if native['captured']==n:return None
    if native['captured']!=n+1 or native['returned']!=n+1:raise AssertionError('Only one last actual capture may be uncommitted')
    value=owned.get('adapter',{}).get('last_capture')
    if value is None:
        if 'last_foundation_capture_return' not in capsules:raise AssertionError('No owned uncommitted capture')
        f=typed_json(capsules['last_foundation_capture_return'])
        if f.pop('__type__',None)!='clock_core.CapturedStep':raise AssertionError('Uncommitted capture type')
        w=f['warnings'];
        if w.pop('__type__',None)!='clock_core.WarningLedger':raise AssertionError('Uncommitted warning type')
    else:
        if value.get('dataclass')!='CapturedStep':raise AssertionError('Adapter partial capture type')
        f=value['fields'];warning=f['warnings']
        if warning.get('dataclass')!='WarningLedger':raise AssertionError('Adapter partial warning type')
        w=warning['fields']
    if set(f)!={'simulation_time','state','torque','warnings'} or set(w)!={'counts','lastinfo'}:raise AssertionError('Partial capture schema')
    if type(f['state']) is not bytes or len(f['state'])!=373*8 or type(f['torque']) is not bytes or len(f['torque'])!=23*8:raise AssertionError('Partial capture byte schema')
    state=np.frombuffer(f['state'],np.float64);command=np.frombuffer(f['torque'],np.float64)
    counts=np.asarray(w['counts'],np.int32);last=np.asarray(w['lastinfo'],np.int32)
    if any(type(v) is not int or not 0<=v<(1<<31) for v in w['counts']) or any(type(v) is not int or not -(1<<31)<=v<(1<<31) for v in w['lastinfo']):raise AssertionError('Native warning integer schema')
    if counts.shape!=(8,) or last.shape!=(8,):raise AssertionError('Native warning width')
    if f['simulation_time']!=state[0]:raise AssertionError('Partial time scalar differs')
    if core['captured']>n:
        if capsules['last_validated_capture_state']!=f['state'] or capsules['last_validated_capture_torque']!=f['torque']:raise AssertionError('Foundation/adapter partial capture differs')
    # Compute strict state truth independently of whether verifier ownership returned.
    q,v,force=state[291:321],state[321:350],state[350:]
    finite=np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(force).all()
    limits=contract['joint_limits']
    strict=bool(finite and np.max(np.maximum(np.maximum(limits[:,0]-q[7:],q[7:]-limits[:,1]),0.))<=1e-6
        and np.max(np.abs(v[6:])/contract['native_velocity'])<=1.
        and np.max(np.abs(force)/contract['native_effort'])<=1+1e-9
        and abs(np.linalg.norm(q[3:7])-1)<=1e-10 and q[2]>=.25
        and np.arccos(np.clip(1-2*(q[4]**2+q[5]**2),-1,1))<=1.2 and not np.any(counts))
    if native['verified']>int(a['step_verified'].sum()) and not strict:raise AssertionError('Native verified credit contradicts owned last state')
    b={k:v.copy() for k,v in a.items()}
    fields=dict(packed_capture=state,step_integration=state[:291],step_qpos=q,step_qvel=v,step_actuator_force=force,
        step_command=command,step_target=a['control_target'][n//10],step_raw_action=a['control_raw_action'][n//10],
        step_warning_counts=counts,step_warning_lastinfo=last,step_expected_simulation_time=np.asarray(state[0],np.float64),step_verified=np.asarray(strict,np.bool_))
    for key,value in fields.items():b[key]=np.concatenate([b[key],value[None]])
    checked=physical.physical_arrays(b,initial,contract)
    return dict(actual_reserved_native_index=n,actual_full291_checked=True,strict_captured_state_passed=strict,
        actual_native_verification_attempted=native['verification_attempts']==native['returned'],
        actual_native_verification_returned=None,verification_return_not_inferred_from_attempt_counter=True,
        actual_foundation_verified=core['verified'],new_verification_credit=0,full_scope_credit=False,
        strict_failures=checked['strict_failures'])
