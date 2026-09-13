"""Explicit fixed-row evidence schema; uncommitted floating entries are NaN."""
import hashlib,json,os,time,shutil
from pathlib import Path
import numpy as np
ROWS=3054
HISTORY_WIDTHS=dict(actions=23,base_ang_vel=3,dof_pos=23,dof_vel=23,projected_gravity=3)
def schema():
    result={}
    def add(name,tail=(),dtype='float64'):result[name]=dict(path=name+'.npy',shape=[ROWS,*tail],dtype=dtype)
    for key in ('dataset','start_control','successor_control','source_frame','successor_frame','center_index','successor_center_index','successor_plan_control','successor_plan_local','successor_plan_accepted_update'):add(key,dtype='int64')
    for key in ('label_valid','nominal_verified','successor_replan_boundary','successor_zero_gain'):add(key,dtype='bool')
    for phase in ('nominal','policy'):
        for key,width in (('qpos',30),('qvel',29)):add(phase+'_'+key,(11,width))
        for key in ('time','expected_time'):add(phase+'_'+key,(11,))
        for key in ('warning_counts','warning_lastinfo'):add(phase+'_'+key,(11,8),'int32')
        for key in ('command_torque','actuator_force'):add(phase+'_'+key,(10,23))
        for key in ('start_integration','end_integration'):add(phase+'_'+key,(291,))
        for key in ('valid_steps','attempted_steps','returned_steps','status'):add(phase+'_'+key,dtype='int32')
    for key in ('incoming_raw_prior','nominal_base_action','policy_head_delta','outgoing_raw_prior','policy_actual_normalized_action','endpoint_base_action','endpoint_head_delta'):add(key,(23,),'float32')
    for key in ('nominal_base_target','nominal_target','policy_raw_target','policy_applied_target','endpoint_base_target','label_fixed_map_target','label_residual_rad','teacher_feedback_raw','teacher_feedback_correction','teacher_preclip_target','endpoint_raw_proposal','endpoint_applied_target'):add(key,(23,))
    for key in ('policy_native_target_clipped','teacher_feedback_clipped','teacher_native_clipped'):add(key,(23,),'bool')
    for key in ('nominal_features','endpoint_features'):add(key,(1069,),'float32')
    add('nominal_state',(52,),'float32');add('endpoint_state',(52,),'float32');add('endpoint_latent',(1,256),'float32')
    add('teacher_tangent',(58,))
    for key in ('nominal_head_output','endpoint_head_output','endpoint_actor_output'):add(key,(1,23),'float32')
    for phase in ('incoming','advanced'):
        add(phase+'_history',(300,),'float32')
        for key,width in HISTORY_WIDTHS.items():add(phase+'_history_'+key,(4,width),'float32')
    return result
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def replace_with_sharing_retry(temp,path):
    # Filesystem metadata retry only: same fully written file, no graph/native call.
    for attempt in range(41):
        try:temp.replace(path);return
        except PermissionError:
            if attempt==40:raise
            time.sleep(.05)

def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    replace_with_sharing_retry(tmp,path)
class Store:
    def __init__(self,folder,resume_directory=None):
        self.folder=Path(folder);self.folder.mkdir(exist_ok=False);self.definition=schema();self.arrays={}
        if resume_directory is not None:
            prior=Path(resume_directory);manifest=json.loads((prior/'manifest.json').read_text())
            assert manifest['complete'] is False
            for key,spec in self.definition.items():
                previous=manifest['arrays'][key]
                assert all(previous[k]==spec[k] for k in ('path','shape','dtype'))
                assert sha(prior/spec['path'])==previous['sha256']
                shutil.copyfile(prior/spec['path'],self.folder/spec['path'])
                assert sha(self.folder/spec['path'])==previous['sha256']
                self.arrays[key]=np.lib.format.open_memmap(self.folder/spec['path'],mode='r+')
            shutil.copyfile(prior/'schema.json',self.folder/'schema.json')
            return
        for key,spec in self.definition.items():
            a=np.lib.format.open_memmap(self.folder/spec['path'],mode='w+',dtype=spec['dtype'],shape=tuple(spec['shape']))
            a.fill(np.nan if a.dtype.kind=='f' else 0);self.arrays[key]=a
        atomic(self.folder/'schema.json',dict(arrays=self.definition,status_codes={'unattempted':0,'complete_feasible':1,'strict_failure':2,'exception':3},
            row_order='dataset0/1/2 each startcontrol250..1267;3054 fixed rows',
            valid_prefix='Only initial sample and valid_steps subsequent samples belong to an attempted row; other float entries remain NaN. label_valid is independent and set last.'))
    def row(self,row,**values):
        for key,value in values.items():self.arrays[key][row]=value
    def flush(self):
        for value in self.arrays.values():value.flush()
    def manifest(self,metadata):
        self.flush();arrays={key:dict(spec,sha256=sha(self.folder/spec['path'])) for key,spec in self.definition.items()}
        result=dict(metadata,arrays=arrays);atomic(self.folder/'manifest.json',result);return result
