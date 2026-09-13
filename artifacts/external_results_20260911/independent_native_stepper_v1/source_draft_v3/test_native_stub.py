"""Injected stubs only: never imports MuJoCo or constructs a native model."""
import base64
import hashlib
import json
import pickle
from types import SimpleNamespace
import unittest
import numpy as np
from model_identity import FullModelIdentity,CALLBACKS
from native_stepper import NativeStepper,NativeAdapterError
from capture_schema import decode,pack,assess_capture,warning_ledger
from clock_core import CapturedStep,PlantFoundation,Binding,Command
from history import MeasuredHistory
from bfm_observations import BFMHistory,state_and_terms


def fake_model():
    m=SimpleNamespace(nq=30,nv=29,nu=23,njnt=24,nbody=25,ngeom=60,na=0,nmocap=0,neq=0,nuserdata=0,nplugin=0,npluginstate=0,
        opt=SimpleNamespace(timestep=.002,integrator=0,disableflags=0,enableflags=0),
        actuator_trnid=np.c_[np.arange(1,24),np.zeros(23,int)],jnt_qposadr=np.r_[0,np.arange(7,30)],jnt_dofadr=np.r_[0,np.arange(6,29)],
        actuator_dyntype=np.zeros(23,int),actuator_gaintype=np.zeros(23,int),actuator_biastype=np.zeros(23,int),
        actuator_gainprm=np.c_[np.ones(23),np.zeros((23,9))],actuator_gear=np.c_[np.ones(23),np.zeros((23,5))],
        jnt_range=np.r_[np.zeros((1,2)),np.tile([-1.,1.],(23,1))],geom_solref=np.full((60,2),.02),geom_solimp=np.full((60,5),.9),
        body_parentid=np.arange(25),vis=SimpleNamespace(arbitrary_visual=1),stat=SimpleNamespace(meanmass=1.))
    return m


def fake_mjb(model):return pickle.dumps(model.__dict__,protocol=4)  # Stub serializer, not a claim about real MJB.


class FakeData:
    def __init__(self):
        self.integration=np.zeros(291,np.float64)
        self.qpos=self.integration[1:31];self.qpos[2]=.8;self.qpos[3]=1.
        self.qvel=self.integration[31:60];self.ctrl=self.integration[89:112]
        self.qfrc_applied=self.integration[112:141];self.xfrc_applied=self.integration[141:].reshape(25,6)
        self.qfrc_actuator=np.zeros(29,np.float64)
        self.warning=SimpleNamespace(number=np.zeros(8,np.int32),lastinfo=np.zeros(8,np.int32))
    @property
    def time(self):return float(self.integration[0])
    @time.setter
    def time(self,value):self.integration[0]=value


class FakeAPI:
    __version__='3.2.3'
    mjtIntegrator=SimpleNamespace(mjINT_EULER=0)
    mjtState=SimpleNamespace(mjSTATE_INTEGRATION=8191)
    def __init__(self):
        self.calls=[];self.step_calls=0;self.set_calls=0;self.get_calls=0;self.save_calls=0
        self.step_error=False;self.partial_get_error=False;self.short_save=False;self.save_error=False
        self.time_delta=.002;self.post_ctrl_corruption=False;self.callback=None
        for name in CALLBACKS:setattr(self,'get_mjcb_'+name,lambda:self.callback)
    def mj_version(self):return 323
    def mj_versionString(self):return '3.2.3'
    def mj_stateSize(self,m,s):return 291
    def mj_sizeModel(self,m):return len(fake_mjb(m))
    def mj_saveModel(self,m,filename=None,buffer=None):
        assert filename is None and buffer.dtype==np.uint8
        self.save_calls+=1
        if self.save_error:raise RuntimeError('injected serialization failure')
        payload=np.frombuffer(fake_mjb(m),np.uint8)
        if self.short_save:buffer[:-1]=payload[:-1]
        else:buffer[:]=payload
    def mj_setState(self,m,d,v,s):self.calls.append('set');self.set_calls+=1;d.integration[:]=v
    def mj_getState(self,m,d,v,s):
        self.get_calls+=1
        if self.partial_get_error:v[:70]=d.integration[:70];raise RuntimeError('injected partial get')
        v[:]=d.integration
    def mj_forward(self,m,d):self.calls.append('forward');d.integration[60:89]=999.
    def mj_step(self,m,d):
        self.calls.append('step');self.step_calls+=1
        d.time+=self.time_delta;d.integration[60]=123.;d.qfrc_actuator[6:]=d.ctrl
        if self.post_ctrl_corruption:d.ctrl[0]+=1.
        if self.step_error:raise RuntimeError('injected after mutation')


def fixture(initialize=True):
    m,d,api=fake_model(),FakeData(),FakeAPI();payload=fake_mjb(m)
    contract={k:np.ones(23,np.float64) for k in ('kp','kd','native_effort','native_velocity')}
    contract.update(default_q=np.zeros(23,np.float64),joint_limits=m.jnt_range[1:].copy())
    stepper=NativeStepper(m,d,api,contract,payload,hashlib.sha256(payload).hexdigest())
    if initialize:stepper.restore_initial(d.integration.copy(),d.warning.number.copy(),d.warning.lastinfo.copy())
    return stepper,m,d,api


def unpack_evidence(payload):
    def get(v):
        kind=v['type']
        if kind=='builtins.dict':return {get(k):get(x) for k,x in v['items']}
        if kind=='numpy.ndarray':return np.frombuffer(get(v['data']),dtype=v['dtype']).reshape(v['shape']).copy()
        if kind=='builtins.bytes':
            assert not v['truncated'];return base64.b64decode(v['prefix'])
        if kind=='builtins.float':return float.fromhex(v['hex'])
        if kind=='builtins.NoneType':return None
        return v['value']
    return get(json.loads(payload))


class IdentityTests(unittest.TestCase):
    def test_complete_contact_and_options_mutations_rejected(self):
        for mutate in (lambda m:m.geom_solref.__setitem__((0,0),.3),lambda m:m.geom_solimp.__setitem__((0,0),.3),
                       lambda m:setattr(m.opt,'disableflags',1),lambda m:setattr(m.opt,'enableflags',1),
                       lambda m:m.body_parentid.__setitem__(2,0),lambda m:setattr(m.vis,'arbitrary_visual',2)):
            m=fake_model();p=fake_mjb(m);mutate(m);api=FakeAPI()
            with self.assertRaises(ValueError):FullModelIdentity(m,api,p,hashlib.sha256(p).hexdigest())
            self.assertEqual(api.step_calls,0)
    def test_wrong_expected_hash_rejected_before_serialization(self):
        m=fake_model();api=FakeAPI()
        with self.assertRaises(ValueError):FullModelIdentity(m,api,fake_mjb(m),'0'*64)
        self.assertEqual(api.save_calls,0)
    def test_short_serialization_detected_by_different_fills(self):
        m=fake_model();api=FakeAPI();api.short_save=True;p=fake_mjb(m)
        with self.assertRaisesRegex(ValueError,'incomplete'):FullModelIdentity(m,api,p,hashlib.sha256(p).hexdigest())
        self.assertEqual(api.save_calls,2)
    def test_wrong_version_and_plugins_and_callbacks(self):
        for mode in ('version','plugin','callback'):
            m=fake_model();api=FakeAPI()
            if mode=='version':api.__version__='3.3.0'
            if mode=='plugin':m.nplugin=1
            if mode=='callback':api.callback=object()
            p=fake_mjb(m)
            with self.assertRaises(ValueError):FullModelIdentity(m,api,p,hashlib.sha256(p).hexdigest())
            self.assertEqual(api.save_calls,0)
    def test_identity_exit_required_and_closes_adapter(self):
        s,m,d,a=fixture();self.assertFalse(s.counters()['model_identity']['exit_verified'])
        self.assertTrue(s.verify_model_exit()['passed']);self.assertTrue(s.closed)
        with self.assertRaises(NativeAdapterError):s.step(np.zeros(23))
        self.assertEqual(a.step_calls,0)
    def test_changed_exit_latches_failure_without_step(self):
        s,m,d,a=fixture();m.opt.disableflags=1
        with self.assertRaises(NativeAdapterError):s.verify_model_exit()
        self.assertFalse(s.identity.exit_verified);self.assertEqual(a.step_calls,0)
        self.assertEqual(s.failure.reason,'MODEL_IDENTITY_CHANGED')
    def test_serialization_exception_accounting(self):
        s,m,d,a=fixture();before=s.identity.serialization_attempted;a.save_error=True
        with self.assertRaises(NativeAdapterError):s.verify_model_exit()
        self.assertEqual(s.identity.serialization_attempted,before+1)
        self.assertEqual(s.identity.serialization_returned,before)


class NativeStubTests(unittest.TestCase):
    def test_initial_restore_order_and_exact_full_state(self):
        s,m,d,a=fixture(False);state=d.integration.copy();state[60:89]=np.arange(29)
        s.restore_initial(state,d.warning.number.copy(),d.warning.lastinfo.copy())
        self.assertEqual(a.calls,['set','forward','set']);np.testing.assert_array_equal(d.integration,state)
        with self.assertRaises(NativeAdapterError):s.restore_initial(state,d.warning.number.copy(),d.warning.lastinfo.copy())
        self.assertEqual(a.set_calls,2)
    def test_full_capture_layout_torque_force_clock(self):
        s,m,d,a=fixture();target=np.full(23,.1);s.step(target);c=s.capture_step()
        self.assertIsNone(s.verify_step(c));decoded=decode(c.state)
        np.testing.assert_array_equal(decoded.integration,d.integration)
        np.testing.assert_array_equal(decoded.actuator_force,d.qfrc_actuator[6:])
        np.testing.assert_array_equal(np.frombuffer(c.torque,np.float64),np.full(23,.1))
        self.assertEqual(len(c.state),2984);self.assertEqual(s.expected_time,.002)
        self.assertEqual((s.attempted,s.returned,s.captured,s.verified),(1,1,1,1))
    def test_step_exception_preserves_full_state_forces_and_command(self):
        s,m,d,a=fixture();a.step_error=True;target=np.full(23,.1)
        with self.assertRaises(NativeAdapterError):s.step(target)
        raw=unpack_evidence(s.failure.evidence)
        np.testing.assert_array_equal(raw['integration'],d.integration)
        np.testing.assert_array_equal(raw['qfrc_actuator'],d.qfrc_actuator)
        self.assertEqual(raw['selected_target'],target.tobytes());self.assertEqual(raw['selected_command'],d.ctrl.tobytes())
        self.assertTrue(raw['integration_get_returned']);self.assertEqual((s.attempted,s.returned,s.captured),(1,0,0))
        with self.assertRaises(NativeAdapterError):s.step(target)
        self.assertEqual(a.step_calls,1);self.assertEqual(a.set_calls,2)
    def test_partial_failed_get_keeps_available_fields(self):
        s,m,d,a=fixture();a.step_error=True;a.partial_get_error=True
        with self.assertRaises(NativeAdapterError):s.step(np.full(23,.1))
        raw=unpack_evidence(s.failure.evidence)
        np.testing.assert_array_equal(raw['integration'][:70],d.integration[:70])
        self.assertTrue(np.isnan(raw['integration'][70:]).all());self.assertFalse(raw['integration_get_returned'])
        np.testing.assert_array_equal(raw['qpos'],d.qpos);np.testing.assert_array_equal(raw['qfrc_actuator'],d.qfrc_actuator)
        self.assertIn('integration',raw['field_errors']);self.assertEqual(a.step_calls,1)
    def test_capture_mismatch_preserves_returned_no_capture_credit(self):
        s,m,d,a=fixture();a.post_ctrl_corruption=True;s.step(np.full(23,.1))
        with self.assertRaises(NativeAdapterError):s.capture_step()
        raw=unpack_evidence(s.failure.evidence)
        np.testing.assert_array_equal(raw['integration'],d.integration)
        self.assertNotEqual(raw['selected_command'],raw['ctrl'].tobytes())
        self.assertEqual((s.attempted,s.returned,s.capture_attempts,s.captured,s.verified),(1,1,1,0,0))
        self.assertEqual(a.step_calls,1)
    def test_capture_get_failure_has_partial_state_and_no_second_step(self):
        s,m,d,a=fixture();s.step(np.zeros(23));a.partial_get_error=True
        with self.assertRaises(NativeAdapterError):s.capture_step()
        raw=unpack_evidence(s.failure.evidence);self.assertIn('integration',raw['field_errors'])
        self.assertEqual(s.returned,1);self.assertEqual(s.captured,0);self.assertEqual(a.step_calls,1)
    def test_strict_failure_keeps_original_owned_capture_separately(self):
        s,m,d,a=fixture();s.step(np.zeros(23));d.qvel[6]=2.;c=s.capture_step()
        self.assertIn('native_joint_speed',s.verify_step(c))
        self.assertEqual(s.failure.capture_return_evidence,s.last_capture_return)
        self.assertEqual(decode(c.state).qvel[6],2.);self.assertEqual(s.verified,0)
    def test_target_validation_no_native_mutation(self):
        s,m,d,a=fixture();before=d.integration.copy()
        with self.assertRaises(NativeAdapterError):s.step(np.full(23,1.01))
        self.assertEqual(a.step_calls,0);self.assertEqual(s.attempted,0);np.testing.assert_array_equal(d.integration,before)
    def test_external_force_prohibited_and_recorded(self):
        s,m,d,a=fixture();d.xfrc_applied[1,0]=1.
        with self.assertRaises(NativeAdapterError):s.step(np.zeros(23))
        self.assertEqual(unpack_evidence(s.failure.evidence)['xfrc_applied'][1,0],1.)
        self.assertEqual(a.step_calls,0)
    def test_mandatory_capture_verify_before_next_step(self):
        s,m,d,a=fixture();s.step(np.zeros(23))
        with self.assertRaises(NativeAdapterError):s.step(np.zeros(23))
        c=s.capture_step()
        with self.assertRaises(NativeAdapterError):s.step(np.zeros(23))
        s.verify_step(c);self.assertEqual(a.step_calls,1)
    def test_independent_clock_not_resynchronized(self):
        s,m,d,a=fixture();a.time_delta=.003;s.step(np.zeros(23));c=s.capture_step()
        self.assertEqual(s.expected_time,.002);self.assertIn('physics_clock',s.verify_step(c))
    def test_boundary_history_keeps_actual_incoming_action(self):
        s,m,d,a=fixture();history=MeasuredHistory();original=BFMHistory()
        for i in range(6):
            _,terms=s.boundary_snapshot();incoming=np.full(23,float(i),np.float32)
            expected=original.before_update(dict(terms,actions=incoming))
            before,flat,measured,after=history.advance(terms,incoming.tobytes())
            self.assertEqual(flat,expected.tobytes());self.assertNotIn('actions',terms)
            self.assertEqual(dict(after)['actions'],original.data['actions'].tobytes())
        self.assertEqual(a.step_calls,0)


class StrictPredicateTests(unittest.TestCase):
    def capture(self,qvel=0.,force=0.,joint=0.,time=.002):
        d=FakeData();d.qvel[6]=qvel;d.qpos[7]=joint;d.time=time;d.qfrc_actuator[6]=force
        return CapturedStep(time,pack(d.integration,d.qpos,d.qvel,d.qfrc_actuator[6:]),d.ctrl.tobytes(),warning_ledger(d.warning.number,d.warning.lastinfo))
    def assess(self,c):return assess_capture(c,np.tile([-1.,1.],(23,1)),np.ones(23),np.ones(23),.002,0.,1)
    def test_speed_and_effort_edges(self):
        self.assertIsNone(self.assess(self.capture(qvel=1.,force=1.)).issue)
        self.assertIn('native_joint_speed',self.assess(self.capture(qvel=np.nextafter(1.,2.))).issue)
        self.assertIn('native_actuator_effort',self.assess(self.capture(force=1.+2e-9)).issue)
    def test_bound_clock_and_initial_effort_edges(self):
        self.assertIn('native_joint_bound',self.assess(self.capture(joint=1.+2e-6)).issue)
        self.assertIn('physics_clock',self.assess(self.capture(time=.002+2e-10)).issue)
        c=self.capture(force=2.,time=0.)
        self.assertIsNone(assess_capture(c,np.tile([-1.,1.],(23,1)),np.ones(23),np.ones(23),0.,0.,0).issue)
    def test_long_repeated_clock_matches_not_multiplication(self):
        expected=0.
        for _ in range(65000):expected+=.002
        self.assertGreater(abs(expected-130.),1e-10)
        c=self.capture(time=expected)
        a=assess_capture(c,np.tile([-1.,1.],(23,1)),np.ones(23),np.ones(23),float(expected),0.,65000)
        self.assertEqual(a.clock_error_seconds,0.);self.assertIsNone(a.issue)
    def test_foundation_exact_clock_stricter_than_oracle_tolerance(self):
        s,m,d,a=fixture();a.time_delta=.002+1e-12
        class Clock:
            now=0
            def now_ns(self):return self.now
            def wait_until_ns(self,n):self.now=max(self.now,n)
        class Mailbox:
            def poll_once(self):return []
            def try_publish(self,*args):return 'PUBLISHED'
        foundation=PlantFoundation(clock=Clock(),stepper=s,epoch_ns=1000000,steps=1,binding=Binding('stub','0'*64,'1'*64,0),
            initial_command=Command.make('zero','stub',np.zeros(23),np.zeros(23,np.float32)),initial_previous_raw=np.zeros(23,np.float32).tobytes(),
            native_limits=m.jnt_range[1:],window_ids=['0'],results=Mailbox(),jobs=Mailbox())
        self.assertFalse(foundation.tick());self.assertIsNone(s.last_assessment.issue)
        self.assertEqual(foundation.step_records.records()[0].issue,'SIMULATION_CLOCK_MISMATCH')


if __name__=='__main__':unittest.main()
