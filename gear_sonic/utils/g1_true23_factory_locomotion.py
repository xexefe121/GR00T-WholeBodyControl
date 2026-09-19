"""Native leg factory locomotion adapter, simulation only.

Recovered FsmHumanLoco contract: five chronological gyro/gravity/q/dq/scaled
action observations; sine/cosine remapped foot clocks, cadence delta, vx/vy/wz.
Leg network observes twelve real joints. Upper body remains separately commanded.
"""
from pathlib import Path
import sys
import numpy as np
import yaml

IDS=np.r_[np.arange(13),np.arange(15,20),np.arange(22,27)]


class FactoryHumanLoco12:
    def __init__(self,firmware,limits,onnx_path=None,*,model_path=None,profile='blind'):
        firmware=Path(firmware)
        self.cfg=yaml.safe_load((firmware/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml').read_text(encoding='utf-8'))
        root=firmware/'ai_sport_files/ai_sport_8.4.2.222/module/ai_sport/file/unitree/module/ai_sport'
        self.path=Path(model_path) if model_path is not None else root/'policies/human_loco/g1_b_l_ankle_track.mnn'
        profiles={name:value for row in self.cfg['policy_params'] for name,value in row.items()}
        if profile not in profiles:raise ValueError(f'Unknown recovered factory profile: {profile}')
        self.profile=profile;self.profile_parameters=np.asarray(profiles[profile],np.float64)
        if model_path is not None and onnx_path is not None:raise ValueError('Select MNN or ONNX factory model')
        self.onnx=None
        if onnx_path:
            import onnxruntime as ort
            opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
            self.onnx=ort.InferenceSession(str(onnx_path),sess_options=opts,providers=['CPUExecutionProvider'])
        else:
            sys.path.insert(0,str(firmware/'python_deps'))
            import MNN
            self.MNN=MNN
            self.net=MNN.Interpreter(str(self.path));self.session=self.net.createSession({'numThread':1,'backend':'CPU'})
            self.inputs={k:self.net.getSessionInput(self.session,k) for k in ('p_obs','cmd')}
            assert self.inputs['p_obs'].getShape()==(1,5,42)
        self.default=np.asarray(self.cfg['default_joint_q'],np.float32)[IDS]
        self.scale=np.asarray(self.cfg['action_scale'],np.float32)[:12]
        self.kp=np.asarray(self.cfg['joint_kp'])[IDS];self.kd=np.asarray(self.cfg['joint_kd'])[IDS]
        self.limits=np.asarray(limits);self.history=np.zeros((5,42),np.float32)
        self.previous=np.zeros(12,np.float32);self.initialized=False
        self.phase=0.;self.frequency=float(self.profile_parameters[7]);self.duty=float(self.profile_parameters[9])
        # FsmHumanLoco's blind callback has a phase0 standing branch (mode4).
        # This network reproduces a quiet physical hold with that convention.
        self.standing_phase=0.
        self.walking=False
        self.phase_sync=None

    def observation_command(self,qpos,qvel,gravity,velocity_command):
        """Advance received robot history and gait clock, without inference."""
        cfg=self.cfg
        obs=np.r_[qvel[3:6]*cfg['observation_scale_angular_vel'],gravity,
            (qpos[7:19]-self.default[:12])*cfg['observation_scale_dof_pos'],
            qvel[6:18]*cfg['observation_scale_dof_vel'],self.previous].astype(np.float32)
        obs=np.clip(obs,-cfg['observation_clip'],cfg['observation_clip'])
        if not self.initialized:self.history[:]=obs;self.initialized=True
        else:self.history[:-1]=self.history[1:];self.history[-1]=obs
        velocity_command=np.asarray(velocity_command)
        # Separate start/stop thresholds prevent repeated gait restarts while
        # closed-loop position feedback makes small standing corrections.
        magnitude=max(np.linalg.norm(velocity_command[:2]),abs(velocity_command[2]))
        if self.phase_sync is not None and self.phase_sync.motion:magnitude=max(magnitude,.13)
        if self.walking and magnitude<.06:self.walking=False
        elif not self.walking and magnitude>.12:self.walking=True
        standing=not self.walking
        frequency=0. if standing else self.frequency
        self.phase=0. if standing else (self.phase+frequency*.02)%1
        if self.phase_sync is not None:self.phase,frequency=self.phase_sync.advance(self.phase,self.walking)
        phase=(self.phase+np.array([0.,.5]))%1
        phase=np.where(phase<self.duty,phase*.5/self.duty,.5+(phase-self.duty)*.5/(1-self.duty))
        if standing:phase[:]=self.standing_phase
        command=np.r_[np.sin(2*np.pi*phase),np.cos(2*np.pi*phase),(frequency-1.2)*.5,velocity_command*.2].astype(np.float32)
        self.last_command=command
        return self.history,command

    def step(self,qpos,qvel,gravity,velocity_command,upper_target=None):
        _,command=self.observation_command(qpos,qvel,gravity,velocity_command)
        if self.onnx:
            raw=self.onnx.run(['act'],{'p_obs':self.history[None],'cmd':command[None]})[0][0]
        else:
            for name,value in (('p_obs',self.history[None]),('cmd',command[None])):
                tensor=self.MNN.Tensor(tuple(value.shape),self.MNN.Halide_Type_Float,value.ravel(),self.MNN.Tensor_DimensionType_Caffe)
                self.inputs[name].copyFrom(tensor)
            self.net.runSession(self.session)
            raw=np.asarray(self.net.getSessionOutput(self.session,'act').getData(),np.float32)
        if raw.shape!=(12,) or not np.isfinite(raw).all():raise ValueError('invalid factory locomotion action')
        target=self.default.copy()
        target[:12]+=np.clip(raw,-10,10)*self.scale
        if upper_target is not None:target[12:]=np.asarray(upper_target)
        target=np.clip(target,self.limits[:,0]+.06,self.limits[:,1]-.06)
        self.previous=target[:12]-self.default[:12]
        return target


class FactoryLocomotionTeleop:
    """Causal root-following plus upper-body commands through the existing gate.

    This candidate is evaluated against every original foot/leg/hand/head goal;
    velocity-following by itself does not qualify it for full-body teleoperation.
    """
    def __init__(self,firmware,contract,*,model,tasks,standing_qpos,now=0.,onnx_path=None,received_gait=False,factory_command_limits=False,fault_standing_capture=False,factory_model=None,factory_profile='blind',root_velocity_feedback=0.):
        from gear_sonic.utils.g1_true23_causal_receiver import CausalReceiver
        c=dict(contract)
        for k in ('default_q','kp','kd','training_effort','native_effort','native_velocity','joint_limits'):c[k]=np.asarray(c[k])
        self.receiver=CausalReceiver(c,model=model,tasks=tasks,standing_qpos=standing_qpos,now=now)
        self.policy=FactoryHumanLoco12(firmware,c['joint_limits'],onnx_path,model_path=factory_model,profile=factory_profile)
        if received_gait:
            from gear_sonic.utils.g1_true23_received_gait import ReceivedGaitPhase
            bounds=self.policy.profile_parameters[6:9]
            self.policy.phase_sync=ReceivedGaitPhase(bounds)
        self.kp,self.kd=self.policy.kp,self.policy.kd
        self.velocity=np.zeros(3)
        self.root_velocity_feedback=float(root_velocity_feedback)
        if not np.isfinite(self.root_velocity_feedback) or self.root_velocity_feedback<0:
            raise ValueError('Root velocity feedback must be finite and nonnegative')
        self.fault_standing_capture=bool(fault_standing_capture)
        self.standing_captured=False
        self.velocity_limits=np.array([1.2,1.,6.])
        if factory_command_limits:
            profile=self.policy.profile_parameters
            self.velocity_limits=np.maximum(profile[:3],profile[3:6])

    def receive(self,packet,task_position,task_quaternion,now):return self.receiver.receive(packet,task_position,task_quaternion,now)

    def prepare(self,qpos,qvel,now):
        import mujoco
        qpos,qvel=np.asarray(qpos),np.asarray(qvel)
        if qpos.shape!=(30,) or qvel.shape!=(29,) or not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            self.receiver.gate.latch('invalid_robot_observation',now);raise ValueError('invalid measured state')
        ref=self.receiver.reference(now)
        self.last_reference=ref
        if self.policy.phase_sync is not None:self.policy.phase_sync.observe(ref)
        rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,qpos[3:7]);rotation=rotation.reshape(3,3)
        yaw=np.arctan2(rotation[1,0],rotation[0,0]);cy,sy=np.cos(yaw),np.sin(yaw)
        heading=np.array([[cy,-sy],[sy,cy]])
        goal_yaw=np.arctan2(ref['root_rotation'][-1,1,0],ref['root_rotation'][-1,0,0])
        yaw_error=np.arctan2(np.sin(goal_yaw-yaw),np.cos(goal_yaw-yaw))
        xy=ref['root_velocity'][-1,:2]+2.5*(ref['root'][-1,:2]-qpos[:2])
        wanted=np.r_[xy@heading,ref['root_omega'][-1,2]+3*yaw_error]
        if self.root_velocity_feedback:
            # Brake measured momentum before position error changes sign.
            # MuJoCo free-joint angular velocity is local; the reference omega
            # is world-frame. Compare their world vertical components.
            velocity_error=np.r_[(qvel[:2]-ref['root_velocity'][-1,:2])@heading,
                rotation[2]@qvel[3:6]-ref['root_omega'][-1,2]]
            wanted-=self.root_velocity_feedback*velocity_error
        # Received recordings include faster turns than the joystick's default
        # cap. Native effort, joint-speed, and position limits stay unchanged.
        wanted=np.clip(wanted,-self.velocity_limits,self.velocity_limits)
        self.velocity+=np.clip(wanted-self.velocity,[-.24,-.24,-.8],[.24,.24,.8])
        if getattr(self,'fault_standing_capture',False) and self.receiver.gate.fault is not None:
            # Once the generated braking reference is stationary and the robot
            # has reached its quiet acceptance region, stop the gait clock.
            # Keep the full neural balance feedback, PD control and fault latch.
            # Small remaining position corrections otherwise sustain shuffling.
            stationary=(np.max(np.abs(ref['joint_velocity'][-1]))<.005
                and np.linalg.norm(ref['root_velocity'][-1])<.005
                and np.linalg.norm(ref['root_omega'][-1])<.005)
            bound=.05 if self.standing_captured else .04
            tilt=np.arccos(np.clip(rotation[2,2],-1,1))
            capture=(self.receiver.gate.fault is not None and stationary
                and np.linalg.norm(ref['root'][-1,:2]-qpos[:2])<bound
                and abs(yaw_error)<.0872665 and tilt<.15
                and np.linalg.norm(qvel[:2])<.25)
            self.standing_captured=bool(capture)
            if capture:
                self.velocity[:]=0.;self.policy.walking=False;self.policy.phase=0.
        else:self.standing_captured=False
        upper=ref['joint'][-1,12:]+self.kd[12:]/self.kp[12:]*ref['joint_velocity'][-1,12:]
        return qpos,qvel,rotation,upper

    def status(self):
        return dict(mode=self.receiver.mode,epoch=self.receiver.gate.epoch,
            fault=self.receiver.gate.fault,explicit_rearm_required=self.receiver.gate.fault is not None,
            future_reference_frames=0,qualified=False,controller='factory_native_locomotion_received',
            velocity_command=self.velocity.tolist(),velocity_command_limits=self.velocity_limits.tolist(),
            root_velocity_feedback_gain=self.root_velocity_feedback,
            fault_standing_capture_enabled=getattr(self,'fault_standing_capture',False),
            fault_standing_captured=getattr(self,'standing_captured',False))

    def command(self,qpos,qvel,now):
        from gear_sonic.utils.g1_true23_causal_controller import ControllerCommand
        qpos,qvel,rotation,upper=self.prepare(qpos,qvel,now)
        target=self.policy.step(qpos,qvel,-rotation[2],self.velocity,upper)
        return ControllerCommand(target,self.status())

    def commit_applied(self,qpos,qvel,target):self.receiver.commit(qpos,qvel,target)

    def import_native_history(self,control):
        if control==0:return
        c=self.receiver.contract;memory=self.receiver.history;p=self.policy
        default=c['default_q'][:12];scale=.25*c['training_effort'][:12]/c['kp'][:12]
        actions,gyro,q,dq,gravity=memory.terms
        previous=default+actions[:,:12]*scale-p.default[:12]
        valid=min(control,4)
        if control<=4:previous[valid-1]=0
        newest=np.concatenate((gyro*.8,gravity,q[:,:12]+default-p.default[:12],dq[:,:12]*.05,previous),1)
        newest[valid:]=newest[valid-1]
        # step() moves these four rows left and appends the current observation.
        p.history[1:]=newest[::-1]
        p.previous[:]=default+memory.prior[:12]*scale-p.default[:12]
        p.initialized=True

    def warmup(self,qpos,qvel,now,calls=10):
        p=self.policy
        observation_time=getattr(self,'last_observation_time',None)
        state=(p.history.copy(),p.previous.copy(),p.initialized,p.phase,p.walking,self.velocity.copy())
        gait_state=p.phase_sync.state() if p.phase_sync is not None else None
        for _ in range(calls):
            alpha=self.body_goal.alpha if hasattr(self,'body_goal') else None
            self.command(qpos,qvel,now)
            p.history[:],p.previous[:]=state[0],state[1]
            p.initialized,p.phase,p.walking=state[2:5];self.velocity[:]=state[5]
            if alpha is not None:self.body_goal.alpha=alpha
            if gait_state is not None:p.phase_sync.restore(gait_state)
            if hasattr(self,'last_observation_time'):self.last_observation_time=observation_time

    def rearm(self,now,measured_qpos):self.receiver.rearm(now,measured_qpos)
