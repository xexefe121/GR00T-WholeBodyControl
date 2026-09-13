"""Plant-owned one-step native adapter. All native APIs are explicitly injected.

This source has no runtime loader, model constructor, worker, private data copy,
sleep, retry, target clamp, state reset after initialization, or hardware path.
"""
from dataclasses import dataclass
import numpy as np
from clock_core import CapturedStep,validated_capture,owned_evidence
from bfm_observations import state_and_terms
from capture_schema import (STATE_SPEC,INTEGRATION_SIZE,typed_vector,warning_ledger,
                            pack,decode,exact,assess_capture)
from model_identity import FullModelIdentity


@dataclass(frozen=True)
class Fault:
    reason: str
    stage: str
    attempted: int
    returned: int
    captured: int
    detail: str
    evidence: bytes
    capture_return_evidence: bytes | None = None


class NativeAdapterError(RuntimeError):
    pass


class NativeStepper:
    def __init__(self,model,data,api,contract,expected_model_mjb,expected_model_sha256):
        self.model,self.data,self.api=model,data,api
        self.attempted=self.returned=self.captured=self.verified=0
        self.capture_attempts=self.verification_attempts=0
        self.stage='MODEL_CONTRACT';self.failure=None
        self.initialized=False;self.restore_attempted=False;self.closed=False
        self.last_capture=None;self.last_capture_return=None;self.last_assessment=None
        self.command=None;self.target=None;self.initial_time=None;self.expected_time=None
        self._capture_control=None
        if (model.nq,model.nv,model.nu,model.njnt,model.nbody,model.ngeom,model.na,model.nmocap,model.neq,model.nuserdata)!=(30,29,23,24,25,60,0,0,0,0):
            raise ValueError('qualified native23 full291 geometry/topology required')
        if abs(float(model.opt.timestep)-.002)>1e-15 or model.opt.integrator!=api.mjtIntegrator.mjINT_EULER:
            raise ValueError('native2ms Euler integration required')
        if int(api.mjtState.mjSTATE_INTEGRATION)!=STATE_SPEC or api.mj_stateSize(model,api.mjtState.mjSTATE_INTEGRATION)!=INTEGRATION_SIZE:
            raise ValueError('exact8191/full291 integration schema required')
        self.identity=FullModelIdentity(model,api,expected_model_mjb,expected_model_sha256)
        self.model_sha256=expected_model_sha256
        if not np.array_equal(model.actuator_trnid[:,0],np.arange(1,24)) or not np.array_equal(model.jnt_qposadr[1:],np.arange(7,30)) or not np.array_equal(model.jnt_dofadr[1:],np.arange(6,29)):
            raise ValueError('native joint/actuator ordering changed')
        if np.any(model.actuator_dyntype) or np.any(model.actuator_gaintype) or np.any(model.actuator_biastype):
            raise ValueError('stateless fixed-gain native torque actuators required')
        if not np.array_equal(model.actuator_gainprm[:,0],np.ones(23)) or not np.array_equal(model.actuator_gear[:,0],np.ones(23)) or np.any(model.actuator_gear[:,1:]):
            raise ValueError('native unit torque gain/gear required')
        self.contract={}
        for key in ('kp','kd','native_effort','native_velocity','default_q'):
            value=typed_vector(np.asarray(contract[key]),23,key,finite=True)
            if key!='default_q' and np.any(value<=0):raise ValueError('positive native gain/limit required: '+key)
            value.setflags(write=False);self.contract[key]=value
        self.limits=np.asarray(model.jnt_range[1:]).copy()
        if self.limits.dtype!=np.float64 or self.limits.shape!=(23,2) or not np.isfinite(self.limits).all() or np.any(self.limits[:,0]>self.limits[:,1]):
            raise ValueError('finite ordered unchanged native bounds required')
        if not exact(self.limits,np.asarray(contract['joint_limits'])):raise ValueError('contract/native joint limits differ')
        self.limits.setflags(write=False)
        self.stage='UNINITIALIZED'

    def _get_integration(self):
        value=np.full(INTEGRATION_SIZE,np.nan,np.float64)
        self.api.mj_getState(self.model,self.data,value,self.api.mjtState.mjSTATE_INTEGRATION)
        return value

    def _no_external_forces(self):
        if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise ValueError('external or root assistance forces are prohibited')

    def _fail(self,reason,exc=None,evidence=None):
        if self.failure is None:
            # Fieldwise diagnostic reads only. A read failure must not discard
            # other available fields or grant any returned/capture credit.
            raw=dict(selected_target=self.target,selected_command=self.command,
                     attempted=self.attempted,returned=self.returned,captured=self.captured,
                     integration_get_returned=False,field_errors={})
            vector=np.full(INTEGRATION_SIZE,np.nan,np.float64)
            try:
                self.api.mj_getState(self.model,self.data,vector,self.api.mjtState.mjSTATE_INTEGRATION)
                raw['integration_get_returned']=True
            except Exception as error:raw['field_errors']['integration']=str(error)
            raw['integration']=vector.copy()  # Retain even a partially filled failed read.
            fields={key:(lambda key=key:np.asarray(getattr(self.data,key)).copy()) for key in
                    ('qpos','qvel','ctrl','qfrc_actuator','qfrc_applied','xfrc_applied')}
            fields.update(time=lambda:float(self.data.time),warnings=lambda:np.asarray(self.data.warning.number).copy(),
                          lastinfo=lambda:np.asarray(self.data.warning.lastinfo).copy())
            for key,getter in fields.items():
                try:raw[key]=getter()
                except Exception as error:raw['field_errors'][key]=str(error)
            self.failure=Fault(reason,self.stage,self.attempted,self.returned,self.captured,
                               '' if exc is None else str(exc),owned_evidence(raw),
                               None if evidence is None else bytes(evidence))

    def verify_model_exit(self):
        """Required after the later experiment, including failures; never rearm."""
        if self.closed:raise NativeAdapterError('adapter already closed; no repeated exit or rearm')
        self.closed=True
        try:return self.identity.verify('exit')
        except Exception as exc:
            self._fail('MODEL_IDENTITY_CHANGED',exc)
            raise NativeAdapterError(str(exc)) from exc

    def _warnings_equal(self,counts,lastinfo):
        if not exact(self.data.warning.number,counts) or not exact(self.data.warning.lastinfo,lastinfo):
            raise ValueError('initial warning ledger changed; no reset permitted')

    def _raw_capture(self,command):
        vector=self._get_integration()
        q=typed_vector(self.data.qpos,30,'qpos');dq=typed_vector(self.data.qvel,29,'qvel')
        force=typed_vector(self.data.qfrc_actuator[6:],23,'actual actuator force')
        state=pack(vector,q,dq,force)
        capture=CapturedStep(float(self.data.time),state,bytes(command),
                             warning_ledger(self.data.warning.number.copy(),self.data.warning.lastinfo.copy()))
        if vector[:1].tobytes()!=np.asarray([capture.simulation_time],np.float64).tobytes():
            raise ValueError('capture integration/sidecar time disagreement')
        if vector[89:112].tobytes()!=bytes(command):raise ValueError('captured native ctrl differs from selected PD command')
        return capture

    def restore_initial(self,integration,warning_counts,warning_lastinfo):
        if self.closed or self.failure is not None or self.restore_attempted or self.returned or self.attempted:raise NativeAdapterError('initial restoration requires an open, unfailed adapter and is allowed once; no rearm/reset')
        self.restore_attempted=True
        try:
            self.stage='INITIAL_INPUT'
            value=typed_vector(integration,291,'initial integration',finite=True)
            if value[0]<0:raise ValueError('negative initial simulation clock')
            warning_ledger(warning_counts,warning_lastinfo)
            counts=warning_counts.copy();lastinfo=warning_lastinfo.copy()
            self._no_external_forces();self._warnings_equal(counts,lastinfo)
            self.initial_time=float(value[0]);self.expected_time=self.initial_time
            self.stage='INITIAL_RESTORE'
            self.api.mj_setState(self.model,self.data,value,self.api.mjtState.mjSTATE_INTEGRATION)
            self._warnings_equal(counts,lastinfo);self._no_external_forces()
            self.stage='INITIAL_FORWARD'
            self.api.mj_forward(self.model,self.data)
            self._warnings_equal(counts,lastinfo);self._no_external_forces()
            self.stage='INITIAL_RESTORE_AFTER_FORWARD'
            self.api.mj_setState(self.model,self.data,value,self.api.mjtState.mjSTATE_INTEGRATION)
            self._warnings_equal(counts,lastinfo);self._no_external_forces()
            if not exact(self._get_integration(),value):raise ValueError('full291 initial restore is not byte exact')
            self.stage='INITIAL_STRICT_CHECK'
            capture=self._raw_capture(self.data.ctrl.tobytes())
            self.last_capture_return=owned_evidence(capture)
            validated_capture(capture)
            assessment=assess_capture(capture,self.limits,self.contract['native_velocity'],self.contract['native_effort'],self.expected_time,self.initial_time,0)
            self.initial_capture=capture;self.last_assessment=assessment
            if assessment.issue is not None:raise ValueError('original initial oracle failure: '+assessment.issue)
            self.initialized=True;self.last_capture_return=None;self.stage='READY'
        except Exception as exc:
            self._fail('INITIALIZATION_FAILED',exc,self.last_capture_return)
            raise NativeAdapterError(str(exc)) from exc

    def _ready(self):
        if not self.initialized or self.failure is not None or self.closed:raise NativeAdapterError('native adapter uninitialized, closed or fault latched')

    def boundary_snapshot(self):
        self._ready()
        if self.returned!=self.verified:raise NativeAdapterError('boundary requires last returned step verified')
        self.stage='BOUNDARY_SNAPSHOT'
        vector=self._get_integration()
        q=typed_vector(self.data.qpos,30,'boundary qpos',finite=True)
        dq=typed_vector(self.data.qvel,29,'boundary qvel',finite=True)
        if vector[1:31].tobytes()!=q.tobytes() or vector[31:60].tobytes()!=dq.tobytes():
            raise NativeAdapterError('boundary integration/measured state disagreement')
        # The dummy action is discarded. Foundation supplies the actual incoming
        # activated raw/feedback action when advancing its owned history.
        _,terms=state_and_terms(q[7:],dq[6:],q[3:7],dq[3:6],np.zeros(23,np.float32),self.contract['default_q'])
        terms={key:value.copy() for key,value in terms.items() if key!='actions'}
        self.stage='READY'
        return vector.tobytes(),terms

    def step(self,target):
        self._ready()
        if self.returned!=self.verification_attempts or self.returned!=self.verified:
            raise NativeAdapterError('no next native step before successful capture/verification')
        self.stage='PD_INPUT'
        try:
            target=typed_vector(target,23,'native target',finite=True)
            if np.any(target<self.limits[:,0]) or np.any(target>self.limits[:,1]):raise ValueError('caller target outside native bounds; no clipping')
            self._no_external_forces()
            q=typed_vector(self.data.qpos,30,'pre-step qpos',finite=True)
            dq=typed_vector(self.data.qvel,29,'pre-step qvel',finite=True)
            kp,kd,effort=(self.contract[key] for key in ('kp','kd','native_effort'))
            command=np.minimum(np.maximum(kp*(target-q[7:])-kd*dq[6:],-effort),effort)
            self.target=target.tobytes();self.command=command.tobytes()
            self.last_capture=None;self.last_capture_return=None
            self.data.ctrl[:]=command
            self.stage='NATIVE_STEP_ATTEMPT';self.attempted+=1
            self.api.mj_step(self.model,self.data)
            self.returned+=1
            self.expected_time+=.002
            self.stage='RETURNED_UNCAPTURED'
        except Exception as exc:
            self._fail('NATIVE_STEP_OR_INPUT_FAILED',exc)
            raise NativeAdapterError(str(exc)) from exc

    def capture_step(self):
        self._ready()
        if self.returned!=self.capture_attempts+1:raise NativeAdapterError('one capture attempt per returned step required')
        self.capture_attempts+=1;self.stage='STEP_CAPTURE'
        try:
            capture=self._raw_capture(self.command)
            self.last_capture_return=owned_evidence(capture)
            capture=validated_capture(capture)
            self.last_capture=capture;self.captured+=1
            self._capture_control=(self.returned,self.expected_time)
            self.stage='CAPTURED_UNVERIFIED'
            return capture
        except Exception as exc:
            self._fail('CAPTURE_FAILED',exc,self.last_capture_return)
            raise NativeAdapterError(str(exc)) from exc

    def verify_step(self,capture):
        self._ready()
        if self.verification_attempts+1!=self.returned or self.last_capture!=capture:
            raise NativeAdapterError('verify exactly the current owned returned capture')
        self.verification_attempts+=1;self.stage='STEP_VERIFY'
        step,expected=self._capture_control
        assessment=assess_capture(capture,self.limits,self.contract['native_velocity'],self.contract['native_effort'],expected,self.initial_time,step)
        self.last_assessment=assessment
        if assessment.issue is None:self.verified+=1;self.stage='READY'
        else:self._fail('STRICT_NATIVE_VIOLATION',assessment.issue,self.last_capture_return)
        return assessment.issue

    def counters(self):
        return dict(attempted=self.attempted,returned=self.returned,capture_attempts=self.capture_attempts,
                    captured=self.captured,verification_attempts=self.verification_attempts,verified=self.verified,
                    expected_time=self.expected_time,initial_time=self.initial_time,initialized=self.initialized,
                    mutation_uncertain=self.attempted>self.returned,stage=self.stage,closed=self.closed,model_identity=self.identity.summary())
