"""DRAFT: fixed full-state saved-map generation. Requires a future selected request.

No MuJoCo, ONNX, Torch, BFM, optimization, rollout or policy calls occur here.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import scipy
from direct_features import DirectFeatures
from secant_math import GROUPS,KEPT,SIGNS,axis_metadata,cell_ids,core_functions,perturb,committed_target,validate_state,validate_tangent

ROWS=3057;AXES=58;TOTAL=ROWS*AXES*2

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def local(path):
    s=str(path).replace('\\','/')
    if sys.platform!='win32' and len(s)>2 and s[1]==':':s='/mnt/'+s[0].lower()+s[2:]
    return Path(s)
def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    for attempt in range(41):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==40:raise
            time.sleep(.05)
def exact(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():raise ValueError(label+' differs')
def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def assert_pins(request):
    for p,h in request['input_sha256'].items():
        if sha(local(p))!=h:raise ValueError('input changed: '+p)
    for p,h in request['source_sha256'].items():
        if sha(local(p))!=h:raise ValueError('source changed: '+p)

def run(request_path,dest):
    # No actual request/clearance is created by source preparation.
    request=read(request_path);request_sha=sha(request_path)
    if request['kind']!='selected_one_full58_saved_secant_generation' or request['generation_selected'] is not True:raise ValueError('unselected draft')
    if request['rows']!=ROWS or request['axes']!=AXES or request['signed_rows']!=TOTAL:raise ValueError('fixed scope')
    if sys.platform=='win32' or np.__version__!='1.26.4' or scipy.__version__!=request['scipy_version']:raise ValueError('pinned WSL pure runtime')
    assert_pins(request)
    own=Path(__file__).resolve()
    if str(own) not in {str(local(p).resolve()) for p in request['source_sha256']}:raise ValueError('executing source not pinned')
    for name in ('direct_features.py','secant_math.py'):
        source=own.with_name(name)
        if str(source) not in {str(local(p).resolve()) for p in request['source_sha256']}:raise ValueError('imported helper not pinned')
    p={k:local(v) for k,v in request['paths'].items()}
    for value in p.values():
        if str(value.resolve()) not in {str(local(k).resolve()) for k in request['input_sha256']}:raise ValueError('unpinned consumed role')
    dest.mkdir(exist_ok=False);arrays={};active={};progress={'phase':'centers','centers_verified':0,'overlap_committed':0,'added_committed':0,'model_calls':0,'physics_steps':0}
    try:
        c=load(p['centers']);contract=read(p['contract']);motion=load(p['motion']);original=load(p['original29'])
        builder=DirectFeatures(motion,original,contract);difference,qmul,qexp=core_functions(p['core'])
        limits=np.asarray(contract['joint_limits'],np.float64);caps=np.asarray(contract['native_velocity'],np.float64)
        radius,group,units=axis_metadata(caps);phase,cell=cell_ids(c['dataset'],c['control'])
        exact(c['joint_limits'],limits,'limits');exact(c['joint_span'],np.diff(limits,axis=1).ravel().astype(np.float32),'span')
        if c['features'].shape!=(ROWS,1069) or c['gain'].shape!=(ROWS,23,58):raise ValueError('center schema')
        if not np.array_equal(c['source_frame'],c['control']+11):raise ValueError('fixed goal clocks')
        # Fixed radii, strict bounds: reject the whole attempt rather than silently shrink/drop slots.
        margins=np.minimum(c['qpos'][:,7:]-limits[:,0],limits[:,1]-c['qpos'][:,7:])
        if not np.all(margins>.01):raise ValueError('fixed joint-position radius not strictly admissible')
        if not np.all(np.abs(c['qvel'][:,6:])+.01*caps<caps):raise ValueError('old velocity radius inadmissible')
        if not np.all(c['qpos'][:,2]>.001):raise ValueError('root-height radius inadmissible')
        nominal_features=[]
        for i in range(ROWS):
            q,v=c['qpos'][i],c['qvel'][i]
            active={'phase':np.asarray('center'),'center':np.int64(i),'qpos':q.copy(),'qvel':v.copy()}
            validate_state(q,v,limits,caps)
            x=builder(q,v,int(c['source_frame'][i]));active['features']=x.copy()
            target,raw,correction,preclip=committed_target(difference,c['planned_state'][i],c['planned_target'][i],c['gain'][i],q,v,limits)
            active.update(target=target,raw=raw,correction=correction,preclip=preclip)
            exact(x,c['features'][i,KEPT],'nominal1000 feature')
            exact(target,c['expert_target'][i],'nominal full map')
            nominal_features.append(x);progress['centers_verified']=i+1
        nominal_features=np.asarray(nominal_features)
        np.savez_compressed(dest/'centers.npz',features=nominal_features,target=c['expert_target'],qpos=c['qpos'],qvel=c['qvel'],
            dataset=c['dataset'],control=c['control'],source_frame=c['source_frame'],phase=phase,cell=cell,
            axis_radius=radius,axis_group=group,axis_units=units,joint_span=c['joint_span'],joint_limits=limits)
        write(dest/'center_gate.json',{'passed':True,'centers_verified':ROWS,'center_sha256':sha(dest/'centers.npz'),'new_probes_attempted':0})
        specs={'features':(1000,np.float32),'target':(23,np.float64),'target_change':(23,np.float64),
            'qpos':(30,np.float64),'qvel':(29,np.float64),'tangent':(58,np.float64),'tangent_change':(58,np.float64),
            'feedback_raw':(23,np.float64),'feedback_correction':(23,np.float64),'preclip':(23,np.float64),
            'feedback_clipped':(23,np.bool_),'native_clipped':(23,np.bool_),'signed_radius':(None,np.float64),'status':(None,np.uint8)}
        for key,(width,dtype) in specs.items():
            array=np.lib.format.open_memmap(dest/(key+'.npy'),mode='w+',dtype=dtype,shape=(ROWS,AXES,2)+(() if width is None else (width,)))
            array[:]=np.nan if np.issubdtype(dtype,np.floating) else 0;arrays[key]=array
        # First generate and verify ALL old velocity overlap. No added-axis output until this gate passes.
        old={k:np.load(p[k],mmap_mode='r',allow_pickle=False) for k in ('velocity_features','velocity_target','velocity_raw','velocity_feedback_clipped','velocity_native_clipped','velocity_value')}
        if old['velocity_features'].shape!=(ROWS,23,2,1069) or old['velocity_target'].shape!=(ROWS,23,2,23):raise ValueError('old overlap shape')
        for current_phase,axes in (('overlap',range(35,58)),('added',range(35))):
            progress['phase']=current_phase
            for i in range(ROWS):
                q0,v0=c['qpos'][i],c['qvel'][i]
                for axis in axes:
                    for sign_index,sign in enumerate(SIGNS):
                        active={'phase':np.asarray(current_phase),'center':np.int64(i),'axis':np.int64(axis),'sign_index':np.int64(sign_index),
                            'signed_radius':np.float64(sign*radius[axis]),'nominal_qpos':q0.copy(),'nominal_qvel':v0.copy()}
                        q,v=perturb(q0,v0,c['planned_state'][i],axis,sign,radius[axis],difference,qmul,qexp)
                        active.update(qpos=q.copy(),qvel=v.copy())
                        validate_state(q,v,limits,caps)
                        tangent,delta=validate_tangent(difference,c['planned_state'][i],q0,v0,q,v,axis,sign*radius[axis])
                        x=builder(q,v,int(c['source_frame'][i]));active.update(features=x.copy(),tangent=tangent,tangent_change=delta)
                        target,raw,correction,preclip=committed_target(difference,c['planned_state'][i],c['planned_target'][i],c['gain'][i],q,v,limits)
                        active.update(target=target.copy(),raw=raw.copy(),correction=correction.copy(),preclip=preclip.copy())
                        if np.array_equal(x,nominal_features[i]):raise ValueError('float32 feature aliases nominal center')
                        if sign_index and np.array_equal(x,arrays['features'][i,axis,0]):raise ValueError('opposite signs alias in float32 features')
                        if not all(np.isfinite(value).all() for value in (x,target,raw,preclip)):raise ValueError('nonfinite returned data')
                        if np.max(np.abs(target-c['expert_target'][i]))>.2+1e-12:raise ValueError('fixed clipped-map target change bound')
                        if current_phase=='overlap':
                            j=axis-35
                            exact(x,old['velocity_features'][i,j,sign_index,KEPT],'old reduced feature overlap')
                            for value,key in ((target,'velocity_target'),(raw,'velocity_raw'),(raw!=correction,'velocity_feedback_clipped'),(preclip!=target,'velocity_native_clipped'),(v[6+j],'velocity_value')):
                                exact(value,old[key][i,j,sign_index],'old '+key+' overlap')
                        values={'features':x,'target':target,'target_change':target-c['expert_target'][i],'qpos':q,'qvel':v,'tangent':tangent,'tangent_change':delta,
                            'feedback_raw':raw,'feedback_correction':correction,'preclip':preclip,'feedback_clipped':raw!=correction,'native_clipped':preclip!=target,'signed_radius':sign*radius[axis]}
                        for key,value in values.items():arrays[key][i,axis,sign_index]=value
                        arrays['status'][i,axis,sign_index]=1
                        progress[current_phase+'_committed']+=1
                if (i+1)%25==0:
                    for array in arrays.values():array.flush()
                    write(dest/'progress.json',progress)
            for array in arrays.values():array.flush()
            if current_phase=='overlap':
                if progress['overlap_committed']!=140622:raise ValueError('overlap count')
                write(dest/'overlap_gate.json',{'passed':True,'overlap_rows':140622,'added_probes_attempted':0,'features_and_full_clipped_map_byteexact':True})
        if progress['added_committed']!=213990 or not np.all(arrays['status']==1):raise ValueError('incomplete fixed slots')
        assert_pins(request)
        if sha(request_path)!=request_sha:raise ValueError('request changed')
        write(dest/'report.json',{'complete':True,'request_sha256':request_sha,'centers':ROWS,'signed_rows':TOTAL,'overlap_rows':140622,'new_rows':213990,
            'original_center_and_overlap_exact':True,'all_slots_valid':True,'status_semantics':{'0':'uncommitted','1':'verified'},
            'radii':radius.tolist(),'groups':GROUPS,'cell_center_counts':np.bincount(cell,minlength=9).tolist(),
            'target_change_normalization':'raw float64 target differences retained; no loss/radius/K normalization selected by generator',
            'output_sha256':{f.name:sha(f) for f in sorted(dest.iterdir()) if f.suffix in ('.npy','.npz')},'all_inputs_unchanged':True,
            'model_calls':0,'BFM_calls':0,'physics_steps':0,'optimizer_updates':0,'replans':0,
            'limitation':'Static state bounds only; no contact consistency, perturbation feasibility or closed-loop stability claim.'})
    except BaseException as error:
        flush_errors=[]
        for key,array in arrays.items():
            try:array.flush()
            except BaseException as flush_error:flush_errors.append({'array':key,'error':repr(flush_error)})
        np.savez_compressed(dest/'failure_active.npz',**active)
        write(dest/'failure.json',{'error':repr(error),'progress':progress,'failure_active_sha256':sha(dest/'failure_active.npz'),
            'flush_errors':flush_errors,
            'committed_slots':'status==1 only; overlap phase all centers before added phase; no retry or dropped rows',
            'model_calls':0,'physics_steps':0})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run(Path(a.request),Path(a.output))
