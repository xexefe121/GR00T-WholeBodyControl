"""Offline-only Orin BFM 50 Hz benchmark.  It opens no socket and sends no command."""
import argparse, gc, json, os, platform, struct, sys, time
from pathlib import Path
import numpy as np
import torch
from gear_sonic.utils.g1_true23_bfm_imu_odometry import Native23IMUOdometry
from gear_sonic.utils.g1_true23_bfmzero_inference import BFMHistory, BFMZeroInference, load_contract, state_and_terms
from gear_sonic.utils.g1_true23_bfmzero_stream import DT, Packet, PacketGate, received_goal
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

LOW_DT=0.002; CONTROL=10; SAMPLE=struct.Struct('<d23f23f4f3f3f'); HEADER=struct.Struct('<IIQd')

class Brake:
    def __init__(self, model, step=.100):
        self.limits=np.asarray(model.jnt_range[1:],dtype=np.float64); self.margin=np.zeros(23)
        self.margin[model.joint('left_ankle_roll_joint').id-1]=.0065; self.low=self.limits[:,0]+self.margin; self.high=self.limits[:,1]-self.margin; self.step=step; self.previous=None
    def apply(self, requested, q, dq):
        policy=np.clip(np.asarray(requested),self.limits[:,0],self.limits[:,1]); target=policy.copy(); pred=np.asarray(q)+np.asarray(dq)*.060
        low=(self.margin>0)&(dq<0)&(pred<self.low); high=(self.margin>0)&(dq>0)&(pred>self.high)
        if np.any(low):
            v=np.minimum(policy[low]+self.step,self.high[low]); target[low]=v if self.previous is None else np.clip(v,self.previous[low]-self.step,self.previous[low]+self.step)
        if np.any(high):
            v=np.maximum(policy[high]-self.step,self.low[high]); target[high]=v if self.previous is None else np.clip(v,self.previous[high]-self.step,self.previous[high]+self.step)
        self.previous=target.copy(); return target

def percentile(values):
    a=np.asarray(values,dtype=np.float64); return {k:float(np.percentile(a,p)) for k,p in [('p50',50),('p95',95),('p99',99)]}|{'max':float(a.max())}
def load_flat(path):
    with open(path,'rb') as f:
        magic,version,n,dt=HEADER.unpack(f.read(HEADER.size)); assert (magic,version,dt)==(0x344D4642,1,.002)
        raw=np.frombuffer(f.read(),dtype=np.uint8); assert len(raw)==n*SAMPLE.size
    a=np.empty((n,57),np.float32)
    for i in range(n): a[i]=SAMPLE.unpack_from(raw,i*SAMPLE.size)
    return dict(timestamp_s=a[:,0].astype(np.float64),joint_q=a[:,1:24],joint_dq=a[:,24:47],imu_quat_wxyz=a[:,47:51],gyro_body=a[:,51:54],accel_specific_force_body=a[:,54:57])
def samples(sensor, i): return {k:v[i] for k,v in sensor.items()}
def load_motion(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files if k!='fps'}
def fields(motion,i): return {k:motion[k][i] for k in ('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w')}
def model(root): return prepare_true23_model(root/'data/g1_23dof_rev_1_0.xml',root/'data/g1_23dof_mujoco_sim2sim.json')[1]
def common(root,threads):
    torch.set_num_threads(threads); torch.set_num_interop_threads(1)
    contract=load_contract(root/'weights/config.yaml'); policy=BFMZeroInference(root/'weights/inference.safetensors','cpu'); m=model(root); sensor=load_flat(root/'data/pico_lowstate_500hz.flatbin'); motion=load_motion(root/'data/pico_teleop_50hz.npz')
    with torch.inference_mode(): policy.actor(torch.zeros((1,52)),torch.zeros((1,23)),torch.zeros((1,300)),torch.zeros((1,256)))
    return contract,policy,m,sensor,motion
def run_path(root,threads,timed=False):
    c,p,m,sensor,motion=common(root,threads); est=Native23IMUOdometry(m); gate=PacketGate(c,stale_seconds=.1); hist=BFMHistory(); action=np.zeros(23,np.float32); brake=Brake(m); result={k:[] for k in ('estimator_batch_ms','admission_ms','feature_goal_ms','inference_ms','brake_ms','total_ms')}; starts=[]; work=[]; misses=0; over=[]; n=min(len(sensor['timestamp_s'])//CONTROL,len(motion['joint_pos'])-11)
    gc0=gc.isenabled(); gc.collect(); gc.freeze(); gc.disable(); priority={}
    if timed:
        try: os.sched_setscheduler(0,os.SCHED_FIFO,os.sched_param(10)); priority['sched_fifo']='applied'
        except Exception as e: priority['sched_fifo']=type(e).__name__+': '+str(e)
        try: os.nice(-10); priority['nice']='applied'
        except Exception as e: priority['nice']=type(e).__name__+': '+str(e)
    start=time.monotonic()
    try:
      for tick in range(n):
        due=start+tick*DT
        if timed:
            while True:
                rem=due-time.monotonic()
                if rem<=0: break
                if rem>.002: time.sleep(rem-.001)
        begun=time.monotonic(); starts.append(max(0.,(begun-due)*1000))
        a=time.perf_counter()
        estimate=None; sample=None
        for j in range(tick*CONTROL,(tick+1)*CONTROL): sample=samples(sensor,j); estimate=est.update(**sample)
        b=time.perf_counter(); result['estimator_batch_ms'].append((b-a)*1000)
        packet=Packet(0,tick,tick*DT,fields(motion,tick+11),tick==n-1); a=time.perf_counter(); accepted=gate.receive(packet,tick*DT); b=time.perf_counter(); result['admission_ms'].append((b-a)*1000)
        if not accepted: raise RuntimeError('gate rejected local recorded packet: '+str(gate.fault))
        a=time.perf_counter(); window=gate.window(); measured=np.r_[estimate['position_start'],sample['imu_quat_wxyz'],sample['joint_q']]; sensed,terms=state_and_terms(sample['joint_q'],sample['joint_dq'],sample['imu_quat_wxyz'],sample['gyro_body'],action,c['default_q']); goal,_,_=received_goal(p,window,measured,1.,2.); history=hist.before_update(terms); b=time.perf_counter(); result['feature_goal_ms'].append((b-a)*1000)
        a=time.perf_counter(); raw=p.actor(torch.as_tensor(sensed[None]),torch.as_tensor(action[None]),torch.as_tensor(history[None]),goal)[0].cpu().numpy(); b=time.perf_counter(); result['inference_ms'].append((b-a)*1000)
        a=time.perf_counter(); action=raw*5.; requested=c['default_q']+action*.25*c['training_effort']/c['kp']; brake.apply(requested,sample['joint_q'],sample['joint_dq']); gate.consume(); b=time.perf_counter(); result['brake_ms'].append((b-a)*1000)
        total=(time.perf_counter()-begun)*1000; result['total_ms'].append(total); work.append(total); misses+=int(time.monotonic()>due+DT)
        if timed and total>18: over.append({'control':tick,'time_s':tick*DT,'work_ms':total})
    finally:
      if gc0: gc.enable()
      gc.unfreeze(); gc.collect()
    return {'controls':n,'components':{k:percentile(v) for k,v in result.items()},'deadline_misses':misses,'start_lateness_ms':percentile(starts),'work_ms':percentile(work),'over_18ms':over,'priority':priority}
def c1(root):
    c,p,m,sensor,motion=common(root,1); ref=np.load(root/'data/windows_c1_reference.npz'); est=Native23IMUOdometry(m); fields_ref=[x for x in ref.files if x.startswith('est_')]; diffs={x[4:]:0. for x in fields_ref}; tick=0
    for i in range(len(ref['est_timestamp_s'])*CONTROL):
      e=est.update(**samples(sensor,i))
      if i%CONTROL==CONTROL-1:
       for k in diffs: diffs[k]=max(diffs[k],float(np.max(np.abs(np.asarray(e[k])-ref['est_'+k][tick]))))
       tick+=1
    actor=[]
    with torch.inference_mode():
      for i in range(200): actor.append(p.actor(torch.from_numpy(ref['state'][i:i+1]),torch.from_numpy(ref['action'][i:i+1]),torch.from_numpy(ref['history'][i:i+1]),torch.from_numpy(ref['goal'][i:i+1]))[0].numpy())
    d=float(np.max(np.abs(np.asarray(actor)-ref['actor_output']))); return {'actor_inputs':200,'actor_max_abs_difference':d,'actor_passed':d<=1e-5,'estimator_controls':tick,'estimator_max_abs_difference_per_field':diffs,'estimator_flag_above_1e6':[k for k,v in diffs.items() if v>1e-6],'passed':d<=1e-5 and not any(v>1e-6 for v in diffs.values())}
def make_reference(root):
    c,p,m,sensor,motion=common(root,1); est=Native23IMUOdometry(m); gate=PacketGate(c,stale_seconds=.1); hist=BFMHistory(); action=np.zeros(23,np.float32); brake=Brake(m); outs={}; actor=[]; state=[]; previous=[]; history=[]; goal=[]; n=min(len(sensor['timestamp_s'])//CONTROL,len(motion['joint_pos'])-11)
    for tick in range(n):
      estimate=None
      for j in range(tick*CONTROL,(tick+1)*CONTROL): sample=samples(sensor,j); estimate=est.update(**sample)
      for k,v in estimate.items(): outs.setdefault('est_'+k,[]).append(np.asarray(v).copy())
      if not gate.receive(Packet(0,tick,tick*DT,fields(motion,tick+11),tick==n-1),tick*DT): raise RuntimeError(gate.fault)
      window=gate.window(); measured=np.r_[estimate['position_start'],sample['imu_quat_wxyz'],sample['joint_q']]; sensed,terms=state_and_terms(sample['joint_q'],sample['joint_dq'],sample['imu_quat_wxyz'],sample['gyro_body'],action,c['default_q']); g,_,_=received_goal(p,window,measured,1.,2.); h=hist.before_update(terms)
      raw=p.actor(torch.as_tensor(sensed[None]),torch.as_tensor(action[None]),torch.as_tensor(h[None]),g)[0].cpu().numpy()
      if len(actor)<200: state.append(sensed); previous.append(action.copy()); history.append(h); goal.append(g.cpu().numpy()[0]); actor.append(raw)
      action=raw*5.; brake.apply(c['default_q']+action*.25*c['training_effort']/c['kp'],sample['joint_q'],sample['joint_dq']);gate.consume()
    np.savez(root/'data/windows_c1_reference.npz',state=np.asarray(state,np.float32),action=np.asarray(previous,np.float32),history=np.asarray(history,np.float32),goal=np.asarray(goal,np.float32),actor_output=np.asarray(actor,np.float32),**{k:np.asarray(v) for k,v in outs.items()})
    return {'reference_controls':n,'actor_inputs':len(actor)}
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('make-reference','c1','b1','b2'));p.add_argument('--root',type=Path,required=True);p.add_argument('--threads',type=int,default=1);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 r=make_reference(a.root) if a.mode=='make-reference' else (c1(a.root) if a.mode=='c1' else run_path(a.root,a.threads,a.mode=='b2')); r|={'mode':a.mode,'threads':a.threads,'python':sys.version,'platform':platform.platform(),'torch':torch.__version__};a.output.write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps(r,indent=2))
if __name__=='__main__': main()
