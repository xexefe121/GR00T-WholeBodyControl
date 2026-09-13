"""One selected12-state current-prior intervention, exactly60 graph calls."""
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
from stateless_adapter import prepare

BASE=Path(__file__).resolve().parent.parent;DEST=BASE/'results'
BUDGET=dict(backward=12,actor=24,head=24)
COUNTS={key:dict(attempted=0,returned=0) for key in BUDGET}
PROGRESS=dict(stage='PREFLIGHT',baseline_completed=0,interventions_completed=0,calls=COUNTS)
ACTIVE={}
def local(p):
    s=str(p).replace('\\','/')
    if sys.platform!='win32' and len(s)>2 and s[1]==':':s='/mnt/'+s[0].lower()+s[2:]
    return Path(s)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def archive(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def atomic_json(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def active_save():
    tmp=DEST/'active.npz.tmp'
    with tmp.open('wb') as f:np.savez_compressed(f,**{k:np.asarray(v) for k,v in ACTIVE.items()});f.flush();os.fsync(f.fileno())
    tmp.replace(DEST/'active.npz')
def exact(a,b,name):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():raise AssertionError('Byte parity: '+name)
def rms(v):return float(np.sqrt(np.mean(np.asarray(v,np.float64)**2)))
def frozen():
    r=read(BASE/'request.json')
    for name,digest in r['source_sha256'].items():assert sha(Path(__file__).parent/name)==digest,name
    for name,digest in r['input_sha256'].items():assert sha(local(name))==digest,name
    return r

class Counted:
    def __init__(self,inner,name):self.inner=inner;self.name=name
    def run(self,outputs,feed):
        count=COUNTS[self.name]
        if count['attempted']>=BUDGET[self.name]:raise RuntimeError('Graph call budget exhausted '+self.name)
        for key,value in feed.items():ACTIVE[self.name+'_input_'+key]=np.asarray(value).copy()
        ACTIVE['active_graph']=np.asarray(self.name);ACTIVE['active_graph_attempt']=np.int64(count['attempted']+1)
        count['attempted']+=1;active_save();atomic_json(DEST/'progress.json',PROGRESS)
        returned=self.inner.run(outputs,feed)
        for i,value in enumerate(returned):ACTIVE[self.name+'_output_'+str(i)]=np.asarray(value).copy()
        count['returned']+=1;active_save();atomic_json(DEST/'progress.json',PROGRESS)
        return returned

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--clearance-sha256',required=True);args=parser.parse_args()
    assert sha(BASE/'clearance.json')==args.clearance_sha256
    clear=read(BASE/'clearance.json');assert clear['approved'] is True
    assert clear['request_sha256']==sha(BASE/'request.json')
    initial_request_sha256=clear['request_sha256'];initial_clearance_sha256=args.clearance_sha256
    assert sha(local(clear['review_path']))==clear['review_sha256']
    request=frozen();paths={k:local(v) for k,v in request['paths'].items()}
    assert request['graph_calls']==BUDGET and request['controls']==list(range(250,262))
    DEST.mkdir(exist_ok=False);atomic_json(DEST/'progress.json',PROGRESS)
    try:
        assert sys.platform!='win32' and np.__version__=='1.26.4'
        sys.path.insert(0,str(paths['onnx_dependencies']))
        import onnxruntime as ort
        assert ort.__version__=='1.23.2'
        binaries=list((Path(ort.__file__).parent/'capi').glob('*pybind11_state*.so'));assert len(binaries)==1
        assert sha(binaries[0]) in request['input_sha256'].values()
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        sessions={name:Counted(ort.InferenceSession(str(paths[name]),sess_options=options,providers=['CPUExecutionProvider']),name) for name in ('actor','backward','head')}
        a=archive(paths['actual']);centers=archive(paths['centers']);maps=archive(paths['fixed_map_arrays']);contract=read(paths['contract'])
        for collection in (a,centers,maps):
            for value in collection.values():value.setflags(write=False)
        exact(maps['control'],np.arange(250,262,dtype=maps['control'].dtype),'fixed map control clock')
        c,builder,latent,sensed=prepare(archive(paths['original']),archive(paths['motion']),archive(paths['original29']),contract,sessions)
        limits=np.asarray(contract['joint_limits']);names=contract['joint_names'];cache=[];baselines=[];interventions=[]
        PROGRESS['stage']='ALL_BASELINES_FIRST'
        def build(control,prior,history,z):
            q,v=a['qpos'][control].copy(),a['qvel'][control].copy()
            state=sensed(q,v,prior);ACTIVE['computed_state']=state.copy()
            returned=sessions['actor'].run(None,dict(state=state[None],last_action=prior[None],history=history[None],z=z))
            assert len(returned)==1 and returned[0].shape==(1,23) and returned[0].dtype==np.float32
            raw=returned[0][0]*5;ACTIVE['base_action']=raw.copy()
            base=c['default_q']+raw*.25*c['training_effort']/c['kp'];ACTIVE['base_target']=base.copy()
            assert np.isfinite(raw).all() and np.isfinite(base).all()
            features=builder(q,v,control+11,base,prior);ACTIVE['computed_features']=features.copy()
            returned=sessions['head'].run(None,dict(features=features[None]))
            assert len(returned)==1 and returned[0].shape==(1,23) and returned[0].dtype==np.float32
            delta=returned[0][0];ACTIVE['delta']=delta.copy();assert np.isfinite(delta).all()
            proposal=base+delta;target=np.clip(proposal,limits[:,0],limits[:,1])
            action=(raw+delta*np.asarray(contract['kp'])/(.25*np.asarray(contract['training_effort']))).astype(np.float32)
            actual=((target-np.asarray(contract['default_q']))*np.asarray(contract['kp'])/(.25*np.asarray(contract['training_effort']))).astype(np.float32)
            result=dict(state=state,base_action=raw,base_target=base,features=features,delta=delta,
                raw_proposal=proposal,target=target,action=action,actual_normalized_action=actual,
                previous_action=prior.copy(),history=history.copy(),latent=z.copy(),qpos=q,qvel=v,control=np.int64(control),source_frame=np.int64(control+11))
            for key,value in result.items():ACTIVE['result_'+key]=np.asarray(value).copy()
            active_save()
            return result
        for control in request['controls']:
            ACTIVE.clear();ACTIVE.update(phase=np.asarray('BASELINE'),control=np.int64(control),qpos=a['qpos'][control].copy(),qvel=a['qvel'][control].copy(),
                prior=a['previous_action'][control].copy(),history=a['history'][control].copy())
            z=latent(control+11,a['qpos'][control]);ACTIVE['latent']=z.copy();assert np.isfinite(z).all()
            baseline=build(control,a['previous_action'][control].copy(),a['history'][control].copy(),z)
            np.savez_compressed(DEST/('baseline_%04d.npz'%control),**ACTIVE)
            for key in ('state','base_target','features','delta','raw_proposal','target','action','actual_normalized_action','previous_action','history'):
                exact(baseline[key],a[key][control],'baseline '+str(control)+' '+key)
            recovered=((a['base_target'][control]-c['default_q'])*c['kp']/(.25*c['training_effort'])).astype(np.float32)
            exact(baseline['base_action'],recovered,'baseline raw actor '+str(control))
            cache.append(z.copy());baselines.append(baseline);PROGRESS['baseline_completed']+=1;atomic_json(DEST/'progress.json',PROGRESS)
        assert PROGRESS['baseline_completed']==12 and COUNTS=={k:dict(attempted=12,returned=12) for k in BUDGET}
        write(DEST/'baseline_parity.json',dict(passed=True,all12_baselines_byteexact=True,counts=COUNTS,
            features_state_base_raw_delta_action_target_exact=True,no_intervention_calls_yet=True))
        PROGRESS['stage']='INTERVENTIONS'
        for i,control in enumerate(request['controls']):
            idx=2038+i;assert centers['dataset'][idx]==2 and centers['control'][idx]==control
            prior=centers['previous_action'][idx].copy();history=a['history'][control].copy();z=cache[i].copy()
            ACTIVE.clear();ACTIVE.update(phase=np.asarray('INTERVENTION'),control=np.int64(control),qpos=a['qpos'][control].copy(),qvel=a['qvel'][control].copy(),
                prior=prior.copy(),history=history.copy(),latent=z.copy(),baseline_previous_action=a['previous_action'][control].copy())
            intervention=build(control,prior,history,z)
            np.savez_compressed(DEST/('intervention_%04d.npz'%control),**ACTIVE)
            for key in ('state','qpos','qvel','history','latent'):exact(intervention[key],baselines[i][key],'unchanged intervention '+key)
            mask=np.ones(1069,bool);mask[52:75]=False;mask[1023:1069]=False
            exact(intervention['features'][mask],baselines[i]['features'][mask],'unchanged feature blocks')
            if control==250:
                for key,value in baselines[i].items():exact(intervention[key],value,'control250 no-op '+key)
            interventions.append(intervention);PROGRESS['interventions_completed']+=1;atomic_json(DEST/'progress.json',PROGRESS)
        assert COUNTS=={k:dict(attempted=v,returned=v) for k,v in BUDGET.items()}
        arrays={}
        for phase,rows in (('baseline',baselines),('intervention',interventions)):
            for key in rows[0]:arrays[phase+'_'+key]=np.asarray([row[key] for row in rows])
        arrays['fixed_map_target']=maps['map_target'];np.savez_compressed(DEST/'arrays.npz',**arrays)
        rows=[]
        for i,(base,changed) in enumerate(zip(baselines,interventions)):
            target=maps['map_target'][i];feedback=maps['feedback_clipped'][i]
            record=dict(control=250+i,fixed_map_feedback_clipped_joints=[names[j] for j in np.flatnonzero(feedback)],
                baseline_raw_vs_map_rmse_rad=rms(base['raw_proposal']-target),baseline_applied_vs_map_rmse_rad=rms(base['target']-target),
                intervention_raw_vs_map_rmse_rad=rms(changed['raw_proposal']-target),intervention_applied_vs_map_rmse_rad=rms(changed['target']-target),
                target_change_rms_rad=rms(changed['target']-base['target']),base_change_rms_rad=rms(changed['base_target']-base['base_target']),
                head_change_rms_rad=rms(changed['delta']-base['delta']),prior_change_rms=rms(changed['previous_action']-base['previous_action']),
                baseline_clipped_joints=[names[j] for j in np.flatnonzero(base['target']!=base['raw_proposal'])],
                intervention_clipped_joints=[names[j] for j in np.flatnonzero(changed['target']!=changed['raw_proposal'])])
            rows.append(record)
        PROGRESS['stage']='FINAL_REHASH';atomic_json(DEST/'progress.json',PROGRESS)
        assert sha(BASE/'request.json')==initial_request_sha256,'Initial request identity changed'
        assert sha(BASE/'clearance.json')==initial_clearance_sha256,'Initial clearance identity changed'
        frozen()
        write(DEST/'report.json',dict(passed=True,all12_baseline_byteexact=True,control250_noop_byteexact=True,
            graph_calls=COUNTS,total_graph_calls=sum(v['returned'] for v in COUNTS.values()),
            request_sha256=sha(BASE/'request.json'),clearance_sha256=sha(BASE/'clearance.json'),arrays_sha256=sha(DEST/'arrays.npz'),
            runtime=dict(numpy=np.__version__,onnxruntime=ort.__version__,binary_sha256=sha(binaries[0]),threads=1,execution='ORT_SEQUENTIAL',provider='CPUExecutionProvider'),
            rows=rows,all_frozen_inputs_unchanged=True,physics_steps=0,optimizer_updates=0,new_training_labels=0,
            limitation='Current-memory input intervention only: actual state and four previous actor-history samples held fixed. Teacher current prior is not physically coherent with the altered student trajectory. Fixed committed-map targets do not replan or certify perturbed states.'))
        PROGRESS['stage']='COMPLETE';atomic_json(DEST/'progress.json',PROGRESS)
        print(json.dumps(dict(passed=True,total_graph_calls=60,rows=rows[:3])),flush=True)
    except BaseException as error:
        active_save();PROGRESS['failed_stage']=PROGRESS['stage'];PROGRESS['stage']='FAILED'
        atomic_json(DEST/'progress.json',PROGRESS)
        write(DEST/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),progress=PROGRESS,
            active_sha256=sha(DEST/'active.npz'),committed_prefix='Only first baseline_completed and interventions_completed per-phase rows passed all gates. Other saved row/active arrays are uncommitted failure evidence.'))
        raise
if __name__=='__main__':
    try:main()
    except BaseException as error:
        if not DEST.exists():write(BASE/'preflight_failure.json',dict(error=repr(error),graph_calls=COUNTS,stage='PREFLIGHT',physics_steps=0,optimizer_updates=0))
        raise
