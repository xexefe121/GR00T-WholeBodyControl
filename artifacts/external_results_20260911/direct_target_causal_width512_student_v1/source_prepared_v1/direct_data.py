"""Load only saved qualified absolute labels; no model or native dependencies."""
from pathlib import Path
import hashlib
import json
import numpy as np
from direct_contract import COUNTS,PHYSICAL_COUNTS,nominal_cells,reduce_features,exact

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def default_paths(base):
    new=Path(base).parent;gen=new/'velocity_chord_student_v1/generation';prior=new/'one_step_physical_student_v1'
    physical=new/'one_step_policy_branch_collection_resume2969_v1/collection'
    return {k:str(v) for k,v in dict(
        contract=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json'),
        centers=gen/'centers.npz',velocity_features=gen/'features.npy',velocity_target=gen/'teacher_target.npy',
        generation_report=gen/'report.json',generation_review=new/'velocity_chord_completed_data_review_v1/review.json',
        pico=new/'pico_walk002_labels_v1/collection/pico/labels.npz',walk002=new/'pico_walk002_labels_v1/collection/walk002/labels.npz',
        pico_report=new/'pico_walk002_labels_v1/collection/pico/report.json',walk002_report=new/'pico_walk002_labels_v1/collection/walk002/report.json',
        broad_saved_review=new/'broader_labels_independent_v1/saved_array_audit/report.json',
        broad_inference_review=new/'broader_labels_independent_v1/baseline_inference_audit/report.json',
        feasibility=new/'direct_target_feature_feasibility_v1/results/report.json',
        physical_manifest=physical/'data/manifest.json',physical_report=physical/'report.json',physical_request=physical.parent/'request.json',
        physical_review=new/'one_step_branch_combined_data_review_v1/review.json',
        sampled_rows=prior/'fit/sampled_center_rows.npy',sampled_axes=prior/'fit/sampled_axes.npy',
        sampler_fit_report=prior/'fit/report.json',sampler_audit=new/'physical_fit_evidence_independent_v1/report.json',
        sampler_audit_request=new/'physical_fit_evidence_independent_v1/request.json').items()}

def array_path(manifest_path,spec):
    root=Path(manifest_path).parent.resolve();path=(root/spec['path']).resolve()
    if path.parent!=root:raise ValueError('Manifest array path escapes its data directory.')
    return path

def collect_input_pins(paths):
    pins={Path(path).as_posix():sha(path) for path in paths.values()}
    manifest=read(paths['physical_manifest'])
    for spec in manifest['arrays'].values():
        path=array_path(paths['physical_manifest'],spec);digest=sha(path)
        if digest!=spec['sha256']:raise ValueError('Physical array hash differs.')
        pins[path.as_posix()]=digest
    return pins

def checked_archive(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in ('features','expert_target','control','source_frame','joint_span','joint_limits') if k in data}

def mapping_to_nominal(center_dataset,center_control,dataset,control):
    identities={(int(d),int(c)):i for i,(d,c) in enumerate(zip(dataset,control))}
    if len(identities)!=len(dataset):raise ValueError('Duplicate nominal dataset/control identity.')
    return np.asarray([identities[(int(d),int(c))] for d,c in zip(center_dataset,center_control)],np.int64)

def load_data(paths,pins):
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Changed bound input: '+path)
    for key,field in [('generation_review','passed'),('physical_review','data_review_pass'),
                      ('broad_saved_review','pass_all'),('broad_inference_review','pass_all'),('sampler_audit','passed')]:
        if read(paths[key])[field] is not True:raise ValueError('Data review not passed: '+key)
    contract=read(paths['contract']);default=np.asarray(contract['default_q'],np.float64);limits=np.asarray(contract['joint_limits'],np.float64)
    centers=checked_archive(paths['centers'])
    with np.load(paths['centers'],allow_pickle=False) as archive:center_dataset=archive['dataset'].copy()
    exact(center_dataset,np.repeat(np.arange(3,dtype=np.int64),1019),'center dataset')
    exact(centers['control'],np.tile(np.arange(250,1269,dtype=np.int64),3),'center controls')
    span=centers['joint_span'];exact(span,(limits[:,1]-limits[:,0]).astype(np.float32),'frozen rounded span')
    exact(centers['joint_limits'],limits,'native limits')
    nominal_x=[reduce_features(centers['features'])];nominal_y=[centers['expert_target']]
    dataset=[center_dataset];control=[centers['control']];frames=[centers['source_frame']];phases=[]
    for d in range(3):phases.append(np.repeat(np.arange(3,dtype=np.int64),COUNTS[d]))
    phase=[np.concatenate(phases)]
    for code,name in ((3,'pico'),(4,'walk002')):
        data=checked_archive(paths[name]);n=sum(COUNTS[code])
        exact(data['joint_span'],span,name+' span');exact(data['joint_limits'],limits,name+' limits')
        exact(data['control'],np.arange(250,250+n,dtype=np.int64),name+' controls')
        exact(data['source_frame'],data['control']+11,name+' frames')
        nominal_x.append(reduce_features(data['features']));nominal_y.append(data['expert_target'])
        dataset.append(np.full(n,code,np.int64));control.append(data['control']);frames.append(data['source_frame'])
        phase.append(np.repeat(np.arange(3,dtype=np.int64),COUNTS[code]))
        with np.load(paths[name]) as original:exact(original['phase'],phase[-1],name+' phase')
    x=np.concatenate(nominal_x);target=np.concatenate(nominal_y);dataset=np.concatenate(dataset);control=np.concatenate(control);frame=np.concatenate(frames);phase=np.concatenate(phase)
    if x.shape!=(9904,1000) or target.shape!=(9904,23) or target.dtype!=np.float64:raise ValueError('Nominal schema.')
    if not np.isfinite(x).all() or not np.isfinite(target).all():raise ValueError('Nonfinite nominal data.')
    if not ((target>=limits[:,0])&(target<=limits[:,1])).all():raise ValueError('Nominal target outside native bounds.')
    cells=nominal_cells(dataset,phase)
    exact(frame,control+11,'all nominal source clocks')
    center_map=mapping_to_nominal(center_dataset,centers['control'],dataset,control)
    exact(x[center_map],reduce_features(centers['features']),'explicit center feature map')
    exact(target[center_map],centers['expert_target'],'explicit center target map')
    vf=np.load(paths['velocity_features'],mmap_mode='r',allow_pickle=False);vt=np.load(paths['velocity_target'],mmap_mode='r',allow_pickle=False)
    if vf.shape!=(3057,23,2,1069) or vf.dtype!=np.float32 or vt.shape!=(3057,23,2,23) or vt.dtype!=np.float64:raise ValueError('Velocity schema.')
    velocity_x=reduce_features(vf).reshape(140622,1000);velocity_target=np.asarray(vt).reshape(140622,23)
    if not np.isfinite(velocity_x).all() or not np.isfinite(velocity_target).all():raise ValueError('Nonfinite velocity labels.')
    if not ((velocity_target>=limits[:,0])&(velocity_target<=limits[:,1])).all():raise ValueError('Velocity targets outside native limits.')
    rows=np.load(paths['sampled_rows'],allow_pickle=False);axes=np.load(paths['sampled_axes'],allow_pickle=False)
    if rows.shape!=(5000,576) or rows.dtype!=np.int32 or axes.shape!=rows.shape or axes.dtype!=np.int8:raise ValueError('Sampler schema.')
    if (axes<0).any() or (axes>=23).any() or (rows<0).any() or (rows>=3057).any():raise ValueError('Sampler range.')
    for cell in range(9):
        picked=rows[:,cell*64:(cell+1)*64]
        allowed_centers=np.flatnonzero(np.isin(center_map,cells[cell]))
        if not np.isin(picked,allowed_centers).all():raise ValueError('Sampler cell mismatch.')
    manifest=read(paths['physical_manifest']);report=read(paths['physical_report'])
    if not manifest['complete'] or not report['completed'] or not report['passed']:raise ValueError('Physical collection incomplete.')
    if sha(paths['physical_manifest'])!=report['manifest_sha256']:raise ValueError('Physical manifest binding.')
    if sha(paths['physical_request'])!=report['request_sha256'] or report['request_sha256']!=manifest['request_sha256']:raise ValueError('Physical request binding.')
    arrays={}
    needed=('dataset','start_control','successor_control','center_index','successor_center_index','label_valid','nominal_verified','nominal_status','policy_status',
            'nominal_features','nominal_target','endpoint_features','label_fixed_map_target','nominal_valid_steps','policy_valid_steps',
            'policy_native_target_clipped','teacher_feedback_clipped','teacher_native_clipped','successor_replan_boundary','successor_zero_gain')
    for key in needed:
        spec=manifest['arrays'][key];a=np.load(array_path(paths['physical_manifest'],spec),allow_pickle=False)
        if list(a.shape)!=spec['shape'] or str(a.dtype)!=spec['dtype']:raise ValueError(key+' physical schema')
        arrays[key]=a
    d=np.repeat(np.arange(3,dtype=np.int64),1018);c=np.tile(np.arange(250,1268,dtype=np.int64),3);start=d*1019+c-250
    for key,value in dict(dataset=d,start_control=c,successor_control=c+1,center_index=start,successor_center_index=start+1).items():exact(arrays[key],value,key)
    for key in ('label_valid','nominal_verified'):
        if arrays[key].dtype!=np.bool_ or not arrays[key].all():raise ValueError('All3054 qualified physical rows required.')
    for key in ('nominal_status','policy_status'):
        if not (arrays[key]==1).all():raise ValueError('Physical status.')
    if not (arrays['nominal_valid_steps']==10).all() or not (arrays['policy_valid_steps']==10).all():raise ValueError('Physical transition incomplete.')
    exact(reduce_features(arrays['nominal_features']),x[center_map[start]],'physical original nominal map')
    exact(arrays['nominal_target'],target[center_map[start]],'physical original target map')
    successor=mapping_to_nominal(d,c+1,dataset,control);exact(successor,center_map[start+1],'physical explicit successor map')
    px=reduce_features(arrays['endpoint_features']);pt=arrays['label_fixed_map_target']
    if pt.dtype!=np.float64 or pt.shape!=(3054,23) or not np.isfinite(pt).all():raise ValueError('Physical target schema.')
    if not np.isfinite(px).all() or not ((pt>=limits[:,0])&(pt<=limits[:,1])).all():raise ValueError('Physical features or native targets invalid.')
    pcells=[np.flatnonzero((d==code)&mask) for code in range(3) for mask in ((c+1<350),((c+1>=350)&(c+1<1169)),(c+1>=1169))]
    if tuple(map(len,pcells))!=PHYSICAL_COUNTS:raise ValueError('Physical requested-cell counts.')
    return dict(features=x,target=target,dataset=dataset,control=control,frame=frame,phase=phase,cells=cells,
        center_map=center_map,center_dataset=center_dataset,center_control=centers['control'],
        velocity_features=velocity_x,velocity_target=velocity_target,sampled_rows=rows,sampled_axes=axes,
        physical_features=px,physical_target=pt,physical_successor=successor,physical_cells=pcells,
        physical_dataset=d,physical_control=c+1,physical_flags={k:arrays[k] for k in needed[-5:]},
        default=default,span=span,limits=limits)
