"""Count/capture native calls around unchanged NativeForecast; no extra steps."""
import mujoco,numpy as np
from native_forecast import NativeForecast
class CapturedForecast:
    def __init__(self,model,contract):
        self.model=model;self.engine=NativeForecast(model,contract,maximum_controls=1)
        self.spec=mujoco.mjtState.mjSTATE_INTEGRATION;self.source=mujoco.MjData(model)
        self.native_step=mujoco.mj_step;self.total_attempted=0;self.total_returned=0;self.active=False
        self.attempted=0;self.returned=0;self.captured=0;self.expected=0.;self.context=None
        self.q=np.full((11,30),np.nan);self.v=np.full((11,29),np.nan);self.t=np.full(11,np.nan);self.et=np.full(11,np.nan)
        self.warnings=np.zeros((11,8),np.int32);self.lastinfo=np.zeros((11,8),np.int32)
        self.torque=np.full((10,23),np.nan);self.force=np.full((10,23),np.nan)
        mujoco.mj_step=self._step
    def vector(self,data):
        a=np.empty(291,np.float64);mujoco.mj_getState(self.model,data,a,self.spec);return a
    def _step(self,model,data):
        if not self.active or model is not self.model or data is not self.engine.private:raise RuntimeError('Unexpected native stepping outside selected forecast')
        if self.total_attempted>=61080 or self.attempted>=10:raise RuntimeError('Native step budget exhausted')
        self.attempted+=1;self.total_attempted+=1
        self.native_step(model,data)
        self.returned+=1;self.total_returned+=1;self.expected+=.002
        i=self.returned;self.q[i]=data.qpos;self.v[i]=data.qvel;self.t[i]=data.time;self.et[i]=self.expected
        self.warnings[i]=data.warning.number;self.lastinfo[i]=data.warning.lastinfo
        self.torque[i-1]=data.ctrl;self.force[i-1]=data.qfrc_actuator[6:];self.captured=i
    def run(self,integration,warning,lastinfo,target,expected_time):
        self.attempted=self.returned=self.captured=0;self.expected=float(expected_time)
        for value in (self.q,self.v,self.t,self.et,self.torque,self.force):value.fill(np.nan)
        self.warnings.fill(0);self.lastinfo.fill(0)
        mujoco.mj_setState(self.model,self.source,integration,self.spec)
        self.source.warning.number[:]=warning;self.source.warning.lastinfo[:]=lastinfo
        assert np.array_equal(self.vector(self.source),integration)
        assert float(self.source.time)==self.expected
        self.q[0]=self.source.qpos;self.v[0]=self.source.qvel;self.t[0]=self.source.time;self.et[0]=self.expected
        self.warnings[0]=warning;self.lastinfo[0]=lastinfo
        self.active=True
        try:
            report,views=self.engine.predict(self.source,target,horizon_controls=1)
            owned={key:value.copy() for key,value in views.items()}
            assert report['physics_steps']==self.returned==self.captured
            for key,left,right in [('qpos',owned['physics_qpos'],self.q[:self.captured+1]),('qvel',owned['physics_qvel'],self.v[:self.captured+1]),
                ('time',owned['physics_time'],self.t[:self.captured+1]),('warnings',owned['warning_counts'],self.warnings[:self.captured+1]),
                ('lastinfo',owned['warning_lastinfo'],self.lastinfo[:self.captured+1]),('torque',owned['physics_torque'],self.torque[:self.captured]),
                ('force',owned['physics_actuator_force'],self.force[:self.captured])]:
                assert left.dtype==right.dtype and left.shape==right.shape and left.tobytes()==right.tobytes(),key
            return report,self.evidence()
        finally:self.active=False
    def evidence(self):
        return dict(qpos=self.q.copy(),qvel=self.v.copy(),time=self.t.copy(),expected_time=self.et.copy(),
            warning_counts=self.warnings.copy(),warning_lastinfo=self.lastinfo.copy(),command_torque=self.torque.copy(),actuator_force=self.force.copy(),
            start_integration=self.vector(self.source),end_integration=self.vector(self.engine.private),
            valid_steps=self.captured,attempted_steps=self.attempted,returned_steps=self.returned)
    def close(self):mujoco.mj_step=self.native_step
