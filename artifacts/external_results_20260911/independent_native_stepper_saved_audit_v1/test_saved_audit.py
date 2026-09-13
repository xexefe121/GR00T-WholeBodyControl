"""Synthetic arrays only; no native runtime or task graph imports."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
from saved_math import exact,unpack373,typed_json,repeated_time,expected_boundaries,compare_case,compare_fault
from audit_saved import identity,api_counts,serialization_records


def typed(value):
    if value is None or type(value) in (bool,int,str):return {'type':'builtins.'+type(value).__name__,'value':value}
    if type(value) is float:return {'type':'builtins.float','hex':value.hex()}
    if type(value) is bytes:return {'type':'builtins.bytes','size':len(value),'sha256':hashlib.sha256(value).hexdigest(),'prefix':base64.b64encode(value).decode(),'truncated':False}
    if type(value) is np.ndarray:return {'type':'numpy.ndarray','dtype':value.dtype.str,'shape':list(value.shape),'data':typed(value.tobytes())}
    if type(value) in (dict,list,tuple):
        items=list(value.items()) if type(value) is dict else list(enumerate(value))
        return {'type':'builtins.'+type(value).__name__,'size':len(items),'truncated':False,'items':[[typed(k),typed(v)] for k,v in items]}
    raise TypeError(type(value))
def payload(value):return json.dumps(typed(value)).encode()

def fixture():
    count=3;times=np.concatenate([np.zeros(1,np.float64),repeated_time(0.,count)])
    q=np.zeros((count+1,30),np.float64);q[:,2]=.8;q[:,3]=1.
    dq=np.zeros((count+1,29),np.float64);states=np.zeros((count+1,291),np.float64)
    states[:,0]=times;states[:,1:31]=q;states[:,31:60]=dq
    warnings=np.zeros((count+1,8),np.int32);force=np.zeros((count,23),np.float64)
    z=dict(target=np.zeros((2,23),np.float64),global_control=np.array([0,1],np.int64),physics_substeps=np.array([2,1],np.int64),
        control_integration_before=states[[0,2]],qpos=q[[0,2,3]],qvel=dq[[0,2,3]],physics_qpos=q,physics_qvel=dq,
        physics_torque=force.copy(),physics_actuator_force=force.copy(),physics_time=times,physics_expected_time=times.copy(),
        physics_warning_number=warnings,physics_warning_lastinfo=warnings.copy(),initial_integration=states[0],final_integration=states[-1],integration_state_spec=np.asarray(8191,np.int64))
    packed=np.concatenate([states[1:],q[1:],dq[1:],force],axis=1).view(np.uint8).reshape(count,2984).copy()
    names,boundaries,_=expected_boundaries([('fixture',z)])
    capture=dict(packed_capture=packed,commanded_torque=force.copy(),simulation_time=times[1:].copy(),warning_counts=warnings[1:].copy(),
        warning_lastinfo=warnings[1:].copy(),control=np.array([0,0,1],np.int64),substep=np.array([1,2,1],np.int64),segment=np.full(3,'fixture',dtype='U32'),
        boundary_name=names,boundary_integration=boundaries)
    contract=dict(kp=np.ones(23,np.float64),kd=np.ones(23,np.float64),native_effort=np.full(23,100.,np.float64),native_velocity=np.full(23,30.,np.float64),joint_limits=np.tile(np.array([-1.,1.]),(23,1)))
    return z,capture,contract

class SavedAuditTests(unittest.TestCase):
    def test_decode_offsets_and_owned_copy(self):
        _,c,_=fixture();d=unpack373(c['packed_capture']);exact(d['integration'][:,0],c['simulation_time'],'time')
        self.assertEqual(d['force'].shape,(3,23));before=d['integration'].copy();c['packed_capture'][:]=0;exact(d['integration'],before,'owned bytes')
    def test_reject_capture_schema_and_overlap(self):
        _,c,_=fixture()
        for value in [c['packed_capture'].astype(np.int32),c['packed_capture'][:,:-1]]:
            with self.assertRaises(ValueError):unpack373(value)
        c['packed_capture'][0,291*8]^=1
        with self.assertRaises(AssertionError):unpack373(c['packed_capture'])
    def test_typed_nested_exact(self):
        values={'v':np.array([0.,-0.,1.]),'i':np.arange(8,dtype=np.int32),'bytes':b'abc','seq':(True,None,7),'time':-0.}
        decoded=typed_json(payload(values));exact(decoded['v'],values['v'],'signed zero');self.assertEqual(decoded['time'].hex(),'-0x0.0p+0')
        self.assertEqual(decoded['seq'],values['seq']);self.assertEqual(decoded['bytes'],b'abc')
    def test_typed_corrupt_bytes(self):
        for key,value in [('sha256','0'*64),('size',4),('prefix','!')]:
            v=typed(b'abc');v[key]=value
            with self.assertRaises((ValueError,TypeError)):typed_json(json.dumps(v).encode())
    def test_typed_truncated_unknown_and_array_size(self):
        a=typed(np.ones(291,np.float64));a['shape']=[290]
        for value in [a,{'type':'unknown'},dict(typed(b'abc'),truncated=True),dict(typed(7),value=True)]:
            with self.assertRaises(ValueError):typed_json(json.dumps(value).encode())
    def test_typed_duplicate_dictionary(self):
        v=typed({'a':1});v['items']*=2;v['size']=2
        with self.assertRaises(ValueError):typed_json(json.dumps(v).encode())
    def test_repeated_clock_not_ideal_product(self):
        values=repeated_time(0.,10000)
        self.assertNotEqual(values[-1],10000*.002)
        self.assertEqual(values[0],.002)
    def test_all_synthetic_fields_and_boundaries(self):
        z,c,contract=fixture();report,_=compare_case(c,[('fixture',z)],contract)
        self.assertEqual(report['samples'],3);self.assertEqual(report['boundaries'],4);self.assertEqual(report['strict_failures'],[])
    def test_mismatched_command_and_time_rejected(self):
        for key in ['commanded_torque','simulation_time','warning_counts','control']:
            z,c,contract=fixture();c[key].flat[-1]+=1
            with self.assertRaises(AssertionError):compare_case(c,[('fixture',z)],contract)
    def test_boundary_full291_warmstart_rejected(self):
        z,c,contract=fixture();packed=c['packed_capture'].view(np.float64).reshape(3,373);packed[1,60]=1.
        with self.assertRaises(AssertionError):compare_case(c,[('fixture',z)],contract)
    def test_external_force_rejected(self):
        z,c,contract=fixture();packed=c['packed_capture'].view(np.float64).reshape(3,373);packed[0,112]=1.
        with self.assertRaises(AssertionError):compare_case(c,[('fixture',z)],contract)
    def test_declared_warning_dtype_rejected(self):
        z,c,contract=fixture();c['warning_counts']=c['warning_counts'].astype(np.int64)
        with self.assertRaises(AssertionError):compare_case(c,[('fixture',z)],contract)
    def test_case_names_order_rejected(self):
        z,c,contract=fixture();c['boundary_name']=c['boundary_name'][::-1]
        with self.assertRaises(AssertionError):compare_case(c,[('fixture',z)],contract)
    def test_fault_capsule_and_return_bytes(self):
        z,c,_=fixture();command=c['commanded_torque'][-1]
        fault=dict(selected_target=z['target'][-1].tobytes(),selected_command=command.tobytes(),attempted=3158,returned=3158,captured=3158,
            integration_get_returned=True,field_errors={},integration=z['final_integration'],qpos=z['physics_qpos'][-1],qvel=z['physics_qvel'][-1],ctrl=command,
            qfrc_actuator=np.zeros(29,np.float64),qfrc_applied=np.zeros(29,np.float64),xfrc_applied=np.zeros((25,6),np.float64),
            time=float(c['simulation_time'][-1]),warnings=c['warning_counts'][-1],lastinfo=c['warning_lastinfo'][-1])
        returned={'type':'clock_core.CapturedStep','fields':{'simulation_time':typed(fault['time']),'state':typed(c['packed_capture'][-1].tobytes()),'torque':typed(command.tobytes()),
            'warnings':{'type':'clock_core.WarningLedger','fields':{'counts':typed(tuple([0]*8)),'lastinfo':typed(tuple([0]*8))}}}}
        report=dict(reason='STRICT_NATIVE_VIOLATION',stage='STEP_VERIFY',detail='native_joint_bound',attempted=3158,returned=3158,captured=3158)
        self.assertTrue(compare_fault(payload(fault),json.dumps(returned).encode(),report,z,c)['expected_physical_failure'])
        fault['selected_target']=np.ones(23,np.float64).tobytes()
        with self.assertRaises(AssertionError):compare_fault(payload(fault),json.dumps(returned).encode(),report,z,c)
    def test_api_accounting_rejects_computed_vs_checked_confusion(self):
        value=dict(step_budget=3158,step_attempted=3158,step_returned=3158,serialization_budget=4,serialization_attempted=4,serialization_returned=4,denied_step_calls=0,denied_serialization_calls=0)
        api_counts(value,3158,4);value['step_returned']=3157
        with self.assertRaises(AssertionError):api_counts(value,3158,4)
    def test_entry_exit_identity(self):
        checks=[dict(phase=p,passed=True,actual_sha256='a'*64,serialization_attempted_before=i*2,serialization_returned_before=i*2,
                     serialization_attempted_after=i*2+2,serialization_returned_after=i*2+2) for i,p in enumerate(['entry','exit'])]
        value=dict(expected_sha256='a'*64,entry_verified=True,exit_verified=True,serialization_attempted=4,serialization_returned=4,
                   process_global_identity_proven=False,portable_cross_platform_identity=False,checks=checks)
        identity(value,'a'*64);value['checks'][1]['actual_sha256']='b'*64
        with self.assertRaises(AssertionError):identity(value,'a'*64)
    def test_all_serialization_buffers_and_order(self):
        expected=b'complete model';digest=hashlib.sha256(expected).hexdigest()
        records=[dict(index=i,native_returned=True,file=f'{i:02d}.mjb',dtype='|u1',shape=[len(expected)],bytes=len(expected),sha256=digest) for i in range(8)]
        serialization_records(records,[expected]*8,expected)
        buffers=[expected]*8;buffers[7]=b'changed'
        with self.assertRaises(AssertionError):serialization_records(records,buffers,expected)
        records[0]['native_returned']=False
        with self.assertRaises(AssertionError):serialization_records(records,[expected]*8,expected)
    def test_no_native_adapter_or_model_imports(self):
        forbidden={'mujoco','torch','onnxruntime','native_stepper','capture_schema','clock_core','replay_core','model_identity'}
        for name in ('saved_math.py','audit_saved.py'):
            tree=ast.parse(Path(__file__).with_name(name).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):self.assertTrue(all(v.name.split('.')[0] not in forbidden for v in node.names))
                if isinstance(node,ast.ImportFrom):self.assertNotIn((node.module or '').split('.')[0],forbidden)

if __name__=='__main__':unittest.main()
