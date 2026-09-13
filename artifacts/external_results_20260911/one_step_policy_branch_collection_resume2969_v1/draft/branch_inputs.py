"""Pure recorded-row mapping, original loader/tangent expressions and history shift."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from collection_arrays import HISTORY_WIDTHS,sha
from stateless_adapter import definitions

def local(path):
    value=str(path).replace('\\','/')
    if sys.platform!='win32' and len(value)>2 and value[1]==':':value='/mnt/'+value[0].lower()+value[2:]
    return Path(value)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def archive(path):
    with np.load(path,allow_pickle=False) as data:return {key:data[key].copy() for key in data.files}
def exact(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():raise AssertionError('Byte parity: '+label)
def clock_table():
    result=np.empty(12681);result[0]=0.
    for i in range(12680):result[i+1]=result[i]+.002
    return result
def row_mapping():
    return [(dataset,control,dataset*1019+control-250) for dataset in range(3) for control in range(250,1268)]
def advance_history(centers,index):
    state=centers['state'][index];prior=centers['previous_action'][index]
    terms=dict(actions=prior,base_ang_vel=state[49:52],dof_pos=state[:23],dof_vel=state[23:46],projected_gravity=state[46:49])
    result={}
    for key in HISTORY_WIDTHS:
        source=centers['history_'+key][index];value=source.copy();value[1:]=source[:-1];value[0]=terms[key];result[key]=value
    return result,np.concatenate([result[key].ravel() for key in sorted(result)]).astype(np.float32)
def frozen_native(bundle):
    import mujoco
    scope=definitions(Path(__file__).parent/'frozen/g1_true23_mjbatch_mpc.py',['load_native_bundle'],dict(np=np,mujoco=mujoco,json=json,Path=Path,sha256=sha))
    return scope['load_native_bundle'](bundle,'walk003')
def fixed_map_function():
    scope=definitions(Path(__file__).parent/'frozen/g1_true23_mjbatch_ilqr_core.py',['quat_mul','quat_log','difference'],dict(np=np),method_class='Planner')
    holder=SimpleNamespace(nq=30,nv=29,lin_q=np.r_[0:3,7:30],lin_v=np.r_[0:3,6:29],free=[(0,0)])
    def apply(centers,index,q,v,limits):
        tangent=scope['difference'](holder,centers['planned_state'][index][None],np.r_[q,v][None])[0]
        raw=centers['gain'][index]@tangent
        correction=np.clip(raw,-.1,.1)
        preclip=centers['planned_target'][index]+correction
        target=np.clip(preclip,limits[:,0],limits[:,1])
        return dict(teacher_tangent=tangent,teacher_feedback_raw=raw,teacher_feedback_correction=correction,
            teacher_preclip_target=preclip,label_fixed_map_target=target,
            teacher_feedback_clipped=raw!=correction,teacher_native_clipped=preclip!=target)
    return apply

class Inputs:
    def __init__(self,paths):
        self.centers=archive(paths['centers']);self.traces=[archive(paths['trace'+str(i)]) for i in range(3)]
        self.old=archive(paths['old_snapshots']);self.actual=archive(paths['actual']);self.witness=archive(paths['witness'])
        self.clock=clock_table();self.rows=row_mapping()
        exact(self.old['control'],np.arange(1269,dtype=np.int64),'old snapshot controls')
        assert int(self.old['integration_spec'])==8191
        for key in ('snapshot_native_steps_attempted','snapshot_native_steps_completed','snapshot_native_steps_verified'):
            exact(self.old[key],np.r_[np.full(1268,10,np.int64),np.int64(0)],key)
        for key in ('snapshot_control_was_started','snapshot_control_fully_verified'):
            exact(self.old[key],np.r_[np.ones(1268,bool),False],key)
        for collection in [self.centers,self.old,self.actual,self.witness,*self.traces]:
            for value in collection.values():value.setflags(write=False)
        self.warning_source_dtypes=[]
        for trace in self.traces:
            self.warning_source_dtypes.append(str(self.warnings(trace).dtype))
    @staticmethod
    def warnings(trace):return trace['physics_warning_number'] if 'physics_warning_number' in trace else trace['physics_warning_counts']
    @staticmethod
    def forces(trace):return trace['physics_actuator_force'] if 'physics_actuator_force' in trace else trace['physics_actuator_torque']
    @staticmethod
    def native_warning(value):
        result=np.asarray(value,dtype=np.int32)
        if not np.array_equal(result,value):raise AssertionError('Warning cast loses values')
        return result
    def integration(self,dataset,control):
        return (self.old if dataset==0 else self.traces[dataset])['control_integration_before'][control].copy()
    def expected(self,trace,control):
        start=control*10;slice_state=slice(start,start+11);slice_force=slice(start,start+10)
        return dict(qpos=trace['physics_qpos'][slice_state],qvel=trace['physics_qvel'][slice_state],
            time=trace['physics_time'][slice_state],expected_time=self.clock[slice_state],
            warning_counts=self.native_warning(self.warnings(trace)[slice_state]),
            warning_lastinfo=self.native_warning(trace['physics_warning_lastinfo'][slice_state]),
            command_torque=trace['physics_torque'][slice_force],actuator_force=self.forces(trace)[slice_force])
    def validate_centers(self,builder,sensed,contract):
        centers=self.centers;limits=np.asarray(contract['joint_limits']);mapping=[]
        for row,(dataset,control,index) in enumerate(self.rows):
            trace=self.traces[dataset]
            for idx,c in ((index,control),(index+1,control+1)):
                assert centers['dataset'][idx]==dataset and centers['control'][idx]==c and centers['source_frame'][idx]==c+11
                for center_key,trace_key in [('qpos','qpos'),('qvel','qvel'),('expert_target','target')]:exact(centers[center_key][idx],trace[trace_key][c],f'center {idx} {center_key}')
                exact(centers['qpos'][idx],trace['physics_qpos'][c*10],f'physics q {idx}')
                exact(centers['qvel'][idx],trace['physics_qvel'][c*10],f'physics v {idx}')
                exact(centers['state'][idx],sensed(centers['qpos'][idx],centers['qvel'][idx],centers['previous_action'][idx]),f'pure state {idx}')
                exact(centers['features'][idx],builder(centers['qpos'][idx],centers['qvel'][idx],c+11,centers['base_target'][idx],centers['previous_action'][idx]),f'pure features {idx}')
                flat=np.concatenate([centers['history_'+key][idx].ravel() for key in sorted(HISTORY_WIDTHS)]).astype(np.float32)
                exact(flat,centers['history'][idx],f'named history {idx}')
                assert np.isfinite(centers['expert_target'][idx]).all() and np.all(centers['expert_target'][idx]>=limits[:,0]) and np.all(centers['expert_target'][idx]<=limits[:,1])
            state=self.expected(trace,control)
            exact(state['time'],state['expected_time'],f'fixed clock {row}')
            if dataset==0:
                exact(self.old['qpos'][control],centers['qpos'][index],'old capture q')
                exact(self.old['qvel'][control],centers['qvel'][index],'old capture v')
                exact(self.old['warning_counts'][control],state['warning_counts'][0],'old capture warnings')
                exact(self.old['warning_lastinfo'][control],state['warning_lastinfo'][0],'old capture lastinfo')
            mapping.append((row,dataset,control,index))
        return mapping
