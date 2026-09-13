"""Header/schema inspection and selective numeric loads, with no feature/map calls."""
import zipfile
from pathlib import Path
import numpy as np

CENTERS={'features':((3057,1069),'float32'),'expert_target':((3057,23),'float64'),
    'dataset':((3057,),'int64'),'control':((3057,),'int64'),'source_frame':((3057,),'int64'),
    'qpos':((3057,30),'float64'),'qvel':((3057,29),'float64'),'planned_state':((3057,59),'float64'),
    'planned_target':((3057,23),'float64'),'gain':((3057,23,58),'float64'),
    'plan_control':((3057,),'int64'),'plan_local':((3057,),'int64'),
    'joint_limits':((23,2),'float64'),'joint_span':((23,),'float32')}
MOTION={'fps':((1,),'float64'),'joint_pos':((1580,23),'float64'),'joint_vel':((1580,23),'float64'),
    'body_pos_w':((1580,24,3),'float64'),'body_quat_w':((1580,24,4),'float64'),
    'body_lin_vel_w':((1580,24,3),'float64'),'body_ang_vel_w':((1580,24,3),'float64')}
ORIGINAL={'fps':((1,),'float64'),'source_task_position_w':((1580,3,3),'float64'),
    'source_task_quaternion_wxyz':((1580,3,4),'float64')}
NPZ_SPECS={'centers':CENTERS,'motion':MOTION,'original29':ORIGINAL}
NPY_SPECS={'velocity_features':((3057,23,2,1069),'float32'),'velocity_target':((3057,23,2,23),'float64'),
    'velocity_raw':((3057,23,2,23),'float64'),'velocity_feedback_clipped':((3057,23,2,23),'bool'),
    'velocity_native_clipped':((3057,23,2,23),'bool'),'velocity_value':((3057,23,2),'float64')}

def header(stream):
    version=np.lib.format.read_magic(stream)
    shape,fortran,dtype=np.lib.format._read_array_header(stream,version)
    return {'shape':list(shape),'dtype':str(dtype),'fortran_order':bool(fortran),'hasobject':bool(dtype.hasobject)}

def archive_headers(path):
    result={}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith('.npy') or '/' in info.filename or '\\' in info.filename:raise ValueError('unexpected NPZ member')
            key=info.filename[:-4]
            if key in result:raise ValueError('duplicate NPZ member')
            with archive.open(info) as stream:result[key]=header(stream)
            # Do not deserialize even unused object metadata.
            if result[key]['hasobject']:raise ValueError('object dtype NPZ member: '+key)
    return result

def check_spec(actual,spec,label):
    shape,dtype=spec
    if actual['hasobject'] or tuple(actual['shape'])!=shape or actual['dtype']!=dtype:raise ValueError('schema '+label)

def load_numeric(path,spec):
    headers=archive_headers(path)
    for key,want in spec.items():
        if key not in headers:raise ValueError('missing archive key '+key)
        check_spec(headers[key],want,key)
    with np.load(path,allow_pickle=False) as archive:values={k:archive[k].copy() for k in spec}
    for key,value in values.items():
        if not np.isfinite(value).all():raise ValueError('nonfinite archive array '+key)
    return values,headers

def validate_inputs(paths,contract):
    loaded={};schemas={}
    for role,spec in NPZ_SPECS.items():loaded[role],schemas[role]=load_numeric(paths[role],spec)
    for role,spec in NPY_SPECS.items():
        with Path(paths[role]).open('rb') as stream:schemas[role]=header(stream)
        check_spec(schemas[role],spec,role)
        value=np.load(paths[role],mmap_mode='r',allow_pickle=False)
        for start in range(0,len(value),64):
            if not np.isfinite(value[start:start+64]).all():raise ValueError('nonfinite old '+role)
    c=loaded['centers'];motion=loaded['motion'];original=loaded['original29']
    if len(contract['body_names'])!=24 or len(contract['joint_names'])!=23:raise ValueError('native names schema')
    for name in ('left_ankle_roll_link','right_ankle_roll_link'):
        if contract['body_names'].count(name)!=1:raise ValueError('native feet identity')
    if motion['fps'][0]!=50 or original['fps'][0]!=50:raise ValueError('source clock')
    if np.any(c['source_frame']<0) or np.any(c['source_frame']+37>=1580):raise ValueError('future goal slots')
    if not np.array_equal(c['source_frame'],c['control']+11):raise ValueError('center frame clock')
    if not np.array_equal(c['plan_control']+c['plan_local'],c['control']):raise ValueError('committed plan clock')
    for array in (motion['body_quat_w'],original['source_task_quaternion_wxyz']):
        if np.max(np.abs(np.linalg.norm(array,axis=-1)-1))>1e-6:raise ValueError('reference quaternion norms')
    return loaded,schemas
