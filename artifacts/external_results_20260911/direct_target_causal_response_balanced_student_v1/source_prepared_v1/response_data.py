"""Read qualified context arrays unchanged; no context fitting or label generation."""
from pathlib import Path
import numpy as np
from direct_data import read,sha
from direct_contract import RETAINED
from full_state_data import load_full_state_data
from full_state_contract import verify_schedule
from context_data import condition_data
from context_contract import exact

def checked_file(path,pins):
    path=Path(path)
    if pins.get(path.as_posix())!=sha(path):raise ValueError('Unfrozen reused context file: '+str(path))
    return path

def load_response_data(request,pins):
    data=load_full_state_data(request['paths'],request['full_state_paths'],pins)
    paths=request['context_paths'];manifest_path=checked_file(paths['manifest'],pins)
    prior_manifest=read(manifest_path)
    for key,path in paths.items():
        checked_file(path,pins)
        if key!='manifest' and prior_manifest['files'].get(Path(path).name)!=sha(path):
            raise ValueError('Reused context manifest member differs: '+key)
    with np.load(paths['normalization'],allow_pickle=False) as archive:
        normalization={key:archive[key].copy() for key in archive.files}
    for key in ('feature_mean','feature_std'):
        array=normalization[key]
        if array.shape!=(1323,) or array.dtype!=np.float32 or not np.isfinite(array).all():
            raise ValueError('Saved1323 normalization schema: '+key)
    if np.any(normalization['feature_std']<=0):raise ValueError('Positive saved std required.')
    for key,value in [('joint_span',data['span']),('default_q',data['default']),('joint_limits',data['limits'])]:
        exact(normalization[key],value,'reused normalization '+key)
    exact(normalization['context_mean'],normalization['feature_mean'][1000:],'context mean slice')
    exact(normalization['context_std'],normalization['feature_std'][1000:],'context std slice')
    for key,rows in [('nominal_context',9904),('center_context',3057),('physical_context',3054)]:
        array=np.load(paths[key],mmap_mode='r',allow_pickle=False)
        if array.shape!=(rows,323) or array.dtype!=np.float32 or not np.isfinite(array).all():
            raise ValueError('Reused causal context schema: '+key)
        data[key]=array
    exact(data['nominal_context'][data['center_map']],data['center_context'],'reused center context')
    expected=dict(dataset=data['dataset'],control=data['control'],phase=data['phase'],source_frame=data['frame'],
        center_to_nominal=data['center_map'],physical_successor=data['physical_successor'],
        physical_dataset=data['physical_dataset'],physical_control=data['physical_control'],
        axis_group=data['full_state_centers']['axis_group'],axis_radius=data['full_state_centers']['axis_radius'])
    with np.load(paths['data_identities'],allow_pickle=False) as archive:
        if set(archive.files)!=set(expected):raise ValueError('Reused data identity keys differ.')
        for key,value in expected.items():exact(archive[key],value,'reused identity '+key)
    data['context_mean']=normalization['context_mean']
    data['context_proof']=read(paths['context_alignment'])
    if (data['context_proof']['nominal_chronological_history_and_prior_pairs']!=9899
        or data['context_proof']['physical_actual_prior_exact'] is not True
        or data['context_proof']['physical_history_once_shifted_exact'] is not True):
        raise ValueError('Original completed chronology proof required.')
    sc=np.load(paths['schedule_centers'],allow_pickle=False)
    sa=np.load(paths['schedule_axes'],allow_pickle=False)
    verify_schedule(sc,sa,data['full_state_centers']['dataset'],data['full_state_centers']['control'],
                    data['full_state_centers']['axis_group'],updates=3000)
    return condition_data(data,'causal'),normalization,sc,sa

