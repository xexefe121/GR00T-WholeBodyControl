"""Experimental native23 AMP locomotion adapter; simulation only.

May21 configs name the 23DOF robot and omit only its two wrist-roll actions.
The supplied networks include observation normalization. UpdateRlState copies raw
gyro, projected gravity, absolute joint positions, velocities and prior actions.
No factory executable runs; only its public neural-network data are loaded.
"""
from pathlib import Path
import numpy as np
import yaml
from gear_sonic.utils.g1_true23_factory_locomotion import FactoryLocomotionTeleop, IDS

ACTIVE = np.r_[np.arange(17), np.arange(18,22)]


class FactoryAmp21:
    def __init__(self, firmware, limits):
        import sys
        firmware = Path(firmware)
        sys.path.insert(0,str(firmware/'python_deps'))
        import MNN
        self.MNN=MNN
        folder = firmware/'decoded_configs/policies/cpy_run_29dofs'
        self.cfg = yaml.safe_load((folder/'fsm_cfg.yaml').read_text(encoding='utf-8'))
        cfg = yaml.safe_load((folder/'May21_11-38-11_9400/cfg_env.yaml').read_text(encoding='utf-8'))
        if cfg['num_dofs'] != 23 or cfg['dof_activate_idx'] != ACTIVE.tolist():
            raise ValueError('This adapter requires the native23 May21 joint topology')
        self.default = np.asarray(self.cfg['default_dof_pos'], np.float32)[IDS]
        self.kp = np.asarray(self.cfg['kp'], float)[IDS]
        self.kd = np.asarray(self.cfg['kd'], float)[IDS]
        self.scale = float(self.cfg['action_scale'])
        self.limits = np.asarray(limits)
        root = firmware/'ai_sport_files/ai_sport_8.4.2.222/module/ai_sport/file/unitree/module/ai_sport/policies/cpy_run_29dofs'
        self.nets=[]
        for name in ('May21_17-41-56_23950','May21_11-38-11_9400'):
            net=MNN.Interpreter(str(root/name/'actor.mnn'))
            session=net.createSession({'numThread':1,'backend':'CPU'})
            inputs={key:net.getSessionInput(session,key) for key in ('memory','commands','proprioception')}
            shapes = {key:list(value.getShape()) for key,value in inputs.items()}
            if shapes != {'memory':[1,345], 'commands':[1,3], 'proprioception':[1,69]}:
                raise ValueError(f'Unexpected native AMP input contract: {shapes}')
            self.nets.append((net,session,inputs))
        self.history=np.zeros((5,69),np.float32)
        self.previous=np.zeros(21,np.float32)
        self.initialized=False; self.walking=False; self.phase=0.; self.phase_sync=None

    def step(self,qpos,qvel,gravity,velocity_command,upper_target=None):
        obs=np.r_[qvel[3:6],gravity,qpos[7:][ACTIVE],qvel[6:][ACTIVE],self.previous].astype(np.float32)
        if not self.initialized:self.history[:]=obs; self.initialized=True
        else:self.history[:-1]=self.history[1:]; self.history[-1]=obs
        magnitude=max(np.linalg.norm(velocity_command[:2]),abs(velocity_command[2]))
        if self.walking and magnitude<.06:self.walking=False
        elif not self.walking and magnitude>.12:self.walking=True
        command=np.asarray(velocity_command,np.float32).copy()
        if not self.walking:command += np.asarray(self.cfg['stance_cmd_offset'],np.float32)
        net,session,inputs=self.nets[int(self.walking)]
        for key,value in {'memory':self.history.reshape(1,345),'commands':command[None],'proprioception':obs[None]}.items():
            tensor=self.MNN.Tensor(tuple(value.shape),self.MNN.Halide_Type_Float,value.ravel(),self.MNN.Tensor_DimensionType_Caffe)
            inputs[key].copyFrom(tensor)
        net.runSession(session)
        raw=np.asarray(net.getSessionOutput(session,'actor_actions').getData(),np.float32)
        if raw.shape!=(21,) or not np.isfinite(raw).all():raise ValueError('Invalid AMP output')
        target=self.default.copy(); target[ACTIVE]+=self.scale*raw
        if upper_target is not None:target[12:]=upper_target
        target=np.clip(target,self.limits[:,0]+.06,self.limits[:,1]-.06)
        self.previous[:]=(target[ACTIVE]-self.default[ACTIVE])/self.scale
        return target


class FactoryAmpTeleop(FactoryLocomotionTeleop):
    def __init__(self,firmware,contract,**kwargs):
        from gear_sonic.utils.g1_true23_causal_receiver import CausalReceiver
        c=dict(contract)
        for key in ('default_q','kp','kd','training_effort','native_effort','native_velocity','joint_limits'):
            c[key]=np.asarray(c[key])
        self.receiver=CausalReceiver(c,**kwargs)
        self.policy=FactoryAmp21(firmware,c['joint_limits'])
        self.kp,self.kd=self.policy.kp,self.policy.kd
        self.velocity=np.zeros(3)
        self.velocity_limits=np.asarray(self.policy.cfg['init_max_cmd'],float)

    def prepare(self,qpos,qvel,now):
        previous=self.velocity.copy()
        result=super().prepare(qpos,qvel,now)
        # The recovered factory command slew limits are per 50Hz command.
        cfg=self.policy.cfg
        up=np.array([cfg['vx_acc_up_delta'],cfg['vy_acc_up_delta'],cfg['yaw_acc_up_delta']])
        down=np.array([cfg['vx_acc_down_delta'],cfg['vy_acc_down_delta'],cfg['yaw_acc_down_delta']])
        limit=np.where(np.abs(self.velocity)>np.abs(previous),up,down)
        self.velocity[:]=previous+np.clip(self.velocity-previous,-limit,limit)
        return result

    def status(self):
        result=super().status(); result['controller']='factory_native23_amp_received'
        return result

    def import_native_history(self,control):
        raise RuntimeError('Independent-clock AMP history adapter has not been qualified')
