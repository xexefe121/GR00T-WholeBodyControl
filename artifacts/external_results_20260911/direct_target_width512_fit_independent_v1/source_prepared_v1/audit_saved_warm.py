"""Audit saved width81000 evidence. No forwards, gradients, optimizers or physics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_math import phase_indices
from audit_full_state_math import BACKENDS,CORPORA,SIZES,schedule,numerical_comparisons,drift_summary
from audit_context_math import WEIGHT,WIDTHS,source_context,endpoint_context,moments,drift,metrics
from audit_restoration import differences
from audit_graph import audit_graph
from audit_balanced_math import energy_weights,balanced_metrics,ledger_expectations,PRODUCER_WEIGHT_RULE
from audit_release import release_paths
from audit_width_math import rate,width_initial_errors,restoration_fields

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def canonical(path):return Path(path).resolve().as_posix()
def counts(calls,rows):return dict(calls_attempted=calls,calls_returned=calls,calls_synchronized=calls,calls_verified=calls,rows_attempted=rows,rows_returned=rows,rows_verified=rows)

def run(request_path,request_sha,output):
    output.mkdir(exist_ok=False);checks=[];tracked={};stage='request';result=None
    def check(name,value):
        checks.append(dict(name=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    def bind(path,digest=None):
        path=Path(path);key=canonical(path);actual=sha(path)
        if digest is not None:check('hash:'+key,actual==digest)
        if key in tracked:check('stable:'+key,actual==tracked[key])
        tracked[key]=actual;return path
    def exact(name,a,b):
        a,b=np.asarray(a),np.asarray(b)
        check(name,a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes())
    def close(name,a,b,nominal=False):
        a,b=np.asarray(a),np.asarray(b)
        check(name,a.shape==b.shape and np.allclose(a,b,rtol=3e-7 if nominal else 5e-12,atol=1e-14,equal_nan=False))
    def compare(name,a,b):
        if isinstance(b,dict):
            check(name+'.keys',isinstance(a,dict) and set(a)==set(b))
            for k,v in b.items():compare(name+'.'+k,a[k],v)
        elif isinstance(b,list):
            check(name+'.length',isinstance(a,list) and len(a)==len(b))
            for i,v in enumerate(b):compare(name+'.'+str(i),a[i],v)
        elif isinstance(b,float):close(name,a,b,nominal=name.endswith(('normalized_MSE','nominal_objective','weighted_objective')))
        else:check(name,type(a)==type(b) and a==b)
    def tree(name,a,b):check(name,not differences(a,b,name))
    def npy(path):return np.load(bind(path),mmap_mode='r',allow_pickle=False)
    def archive(path):
        with np.load(bind(path),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def literal(subject,path,name):
        check('literal:'+name,canonical(subject['path'])==canonical(path) and subject['sha256']==sha(path));bind(path,subject['sha256'])
    def manifest(folder,path,required):
        value=read(bind(path));check('manifest_required:'+str(folder),set(required)<=set(value['files']))
        for name,digest in value['files'].items():
            p=(folder/name).resolve();check('safe_output:'+name,p.parent==folder.resolve());bind(p,digest)
        check('manifest_request:'+str(folder),value['training_request_sha256']==training_sha and value['frozen_receipt_sha256']==frozen_sha)
        return value
    try:
        bind(request_path,request_sha);ar=read(request_path)
        check('audit_kind',ar['kind']=='saved_width512_warm_only')
        import sys
        check('audit_runtime',canonical(sys.executable)==canonical(ar['python_path']) and np.__version__=='1.23.5')
        review=read(bind(ar['source_review']['path'],ar['source_review']['sha256']))
        check('source_review_pass',review[ar['source_review']['pass_field']] is True)
        for path,digest in ar['source_sha256'].items():
            bind(path,digest);check('source_review:'+Path(path).name,review['source_sha256'][Path(path).name]==digest)
        writer=ar['request_writer'];bind(writer['path'],writer['sha256'])
        check('reviewed_request_writer',Path(writer['path']).name=='prepare_audit_request.py' and review['helper_sha256']['prepare_audit_request.py']==writer['sha256'])
        for role,subject in ar['subjects'].items():bind(subject['path'],subject['sha256'])
        base=Path(ar['experiment']);fit=base/'fit';shared=fit/'shared'
        req_path=base/'training_request.json';frozen_path=base/'training_frozen_inputs.json'
        training_sha=sha(bind(req_path));frozen_sha=sha(bind(frozen_path))
        literal(ar['subjects']['training_request'],req_path,'training_request');literal(ar['subjects']['frozen_inputs'],frozen_path,'frozen_inputs')
        request=read(req_path);frozen=read(frozen_path)
        check('frozen_request',frozen['training_request_sha256']==training_sha)
        for path,digest in frozen['input_sha256'].items():bind(path,digest)
        for name,digest in frozen['source_sha256'].items():bind(Path(frozen['source_directory'])/name,digest)
        expected_budget=dict(training_forward_rows=146860000,training_forward_calls=30000,training_updates=10000,
            diagnostic_Torch_rows=1470280,diagnostic_Torch_calls=5748,diagnostic_ORT_rows=367570,diagnostic_ORT_calls=1437,
            calibration_forward_calls=0,calibration_gradient_calls=0,native_calls=0,BFM_calls=0,manual_export_trace_calls=0)
        for name,value in dict(kind='causal_width512_warm_continuation',root_selected=True,condition='causal',
            updates=10000,ordinary_start_step=71000,ordinary_final_step=81000,optimizer_start_step=6000,optimizer_final_step=16000,
            fresh_optimizer=False,coefficient=WEIGHT,budgets=expected_budget,learning_rate=dict(ramp_updates=250,ramp_inclusive=[1e-6,1e-5],cosine_updates=9750,cosine_inclusive=[1e-5,1e-6]),weight_decay=1e-5,
            gradient_clip=10.,coefficient_recalibration=False,initial_byte_gate_required=False,
            initial_parity_tolerance_rad=1e-5,parity_tolerance_rad=1e-5,features=1323,architecture=[1323,512,512,23],old_hidden_width=256,new_hidden_width=512,expansion_seed=20260912,context_order='previous_action23_then_incoming_history300',first_layer_execution='split_old256_new256_original1000_plus323',export_first_layer_execution='monolithic_float64_1323',automatic_retry=False,no_checkpoint_selection=True,context_and_normalization_reused=True,old_optimizer_blocks_exact=True,new_optimizer_moments_zero=True,shared_optimizer_step_for_new_entries=6000).items():compare('request.'+name,request[name],value)
        clearance=read(bind(base/'training_clearance.json'))
        check('clearance',clearance['approved'] is True and clearance['request_sha256']==training_sha and clearance['frozen_receipt_sha256']==frozen_sha)
        launch_review=read(bind(clearance['review_path'],clearance['review_sha256']));bind(clearance['launcher_path'],clearance['launcher_sha256'])
        check('concrete_review',launch_review['prelaunch_review_pass'] is True and launch_review['training_request_sha256']==training_sha and launch_review['frozen_receipt_sha256']==frozen_sha)
        for field,digest in dict(training_request_sha256=training_sha,frozen_receipt_sha256=frozen_sha,launcher_sha256=clearance['launcher_sha256']).items():check('concrete_literal:'+field,launch_review[field]==digest)
        compare('concrete_source_map',launch_review['source_sha256'],frozen['source_sha256'])
        for role,subject in request['subjects'].items():
            bind(subject['path'],subject['sha256'])
            check('consumed_subject_membership:'+role,frozen['input_sha256'].get(Path(subject['path']).as_posix())==subject['sha256'])
            if 'pass_field' in subject:
                q=read(subject['path']);check('qualification:'+role,q[subject['pass_field']] is True)
                for field,value in subject.get('required_fields',{}).items():compare('qualification:'+role+'/'+field,q[field],value)
        frozen_members={canonical(p):d for p,d in frozen['input_sha256'].items()}
        for group in ('paths','full_state_paths','restoration_predictions','context_paths','schedule_paths'):
            for name,path in request[group].items():check('consumed_path_membership:'+group+'/'+name,frozen_members.get(canonical(path))==sha(bind(path)))
        stage='shared_data_and_context'
        paths=request['paths'];centers=archive(paths['centers']);pico=archive(paths['pico']);walk=archive(paths['walk002'])
        kept=np.r_[0:52,75:1023].astype(np.int64)
        features=np.concatenate([z['features'][:,kept] for z in (centers,pico,walk)])
        teacher=np.concatenate([z['expert_target'] for z in (centers,pico,walk)])
        check('original_nominal_schema',features.shape==(9904,1000) and features.dtype==np.float32 and teacher.shape==(9904,23) and teacher.dtype==np.float64)
        dataset=np.r_[np.repeat(np.arange(3,dtype=np.int64),1019),np.full(5980,3,np.int64),np.full(867,4,np.int64)]
        controls=np.r_[np.tile(np.arange(250,1269,dtype=np.int64),3),np.arange(250,6230,dtype=np.int64),np.arange(250,1117,dtype=np.int64)]
        phases=np.concatenate([np.full(len(ids),i%3,np.int64) for i,ids in enumerate(phase_indices())])
        exact('center_dataset',centers['dataset'],dataset[:3057])
        for name,z,start,stop in [('centers',centers,0,3057),('pico',pico,3057,9037),('walk002',walk,9037,9904)]:
            exact(name+'_control',z['control'],controls[start:stop]);exact(name+'_frame',z['source_frame'],controls[start:stop]+11)
            if name!='centers':exact(name+'_phase',z['phase'],phases[start:stop])
        contract=read(bind(paths['contract']));default=np.asarray(contract['default_q'],np.float64);limits=np.asarray(contract['joint_limits'],np.float64);span=centers['joint_span']
        exact('span',span,np.diff(limits,axis=1).ravel().astype(np.float32))
        context_parts=[];chronology=0
        for name,z in [('centers',centers),('pico',pico),('walk002',walk)]:
            value,n=source_context(z,contract);context_parts.append(value);chronology+=n;check('source_context:'+name,True)
        nominal_context=np.concatenate(context_parts)
        full_path=Path(request['full_state_paths']['manifest']);full_manifest=read(bind(full_path))
        def full_array(name):
            s=full_manifest['arrays'][name];a=npy(bind(full_path.parent/s['path'],s['sha256']))
            check('full_schema:'+name,list(a.shape)==s['shape'] and str(a.dtype)==s['dtype']);return a
        full_centers=archive(full_path.parent/'centers.npz')
        exact('full_centers_features',full_centers['features'],features[:3057]);exact('full_centers_target',full_centers['target'],teacher[:3057])
        for name,value in [('dataset',dataset[:3057]),('control',controls[:3057]),('source_frame',controls[:3057]+11),('phase',phases[:3057].astype(np.int8))]:exact('full_center_'+name,full_centers[name],value)
        full_teacher=full_array('target').reshape(354612,23);change=full_array('target_change').reshape(354612,23)
        check('all_full58_valid',bool(np.all(full_array('status')==1)))
        for start in range(0,354612,8192):
            stop=min(start+8192,354612);exact('full_change:'+str(start),change[start:stop],full_teacher[start:stop]-teacher[np.arange(start,stop)//116])
        physical_path=Path(paths['physical_manifest']);physical_manifest=read(bind(physical_path))
        def phys(name):
            s=physical_manifest['arrays'][name];a=npy(bind(physical_path.parent/s['path'],s['sha256']))
            check('physical_schema:'+name,list(a.shape)==s['shape'] and str(a.dtype)==s['dtype']);return a
        physical_dataset=np.repeat(np.arange(3,dtype=np.int64),1018);physical_control=np.tile(np.arange(251,1269,dtype=np.int64),3)
        successor=physical_dataset*1019+physical_control-250;start=successor-1
        for name,value in [('dataset',physical_dataset),('successor_control',physical_control),('successor_center_index',successor),('center_index',start)]:exact('physical_'+name,phys(name),value)
        check('all_physical_valid',bool(np.all(phys('label_valid'))))
        pnames=['center_index','incoming_history','incoming_raw_prior','advanced_history','policy_applied_target','policy_actual_normalized_action','outgoing_raw_prior','policy_native_target_clipped']+['advanced_history_'+name for name,width in WIDTHS]
        pa={name:phys(name) for name in pnames};physical_context=endpoint_context(centers,pa,contract);check('physical_context_reconstruction',True)
        manifest(shared,shared/'output_manifest.json',['normalization.npz','nominal_context.npy','center_context.npy','physical_context.npy','data_identities.npz','context_alignment.json','schedule_centers.npy','schedule_axes.npy','source_schedule_centers.npy','source_schedule_axes.npy','schedule_lineage.json','reused_inputs.json','runtime.json'])
        for name,value in [('nominal_context',nominal_context),('center_context',nominal_context[:3057]),('physical_context',physical_context)]:exact('all_'+name,npy(shared/(name+'.npy')),value)
        alignment=read(shared/'context_alignment.json')
        for name,value in dict(all_nominal_rows_assigned_once=True,nominal_rows=9904,center_rows=3057,physical_rows=3054,
            nominal_chronological_history_and_prior_pairs=chronology,physical_history_once_shifted_exact=True,physical_actual_prior_exact=True,
            physical_applied_prior_differs_raw_rows=int(np.any(pa['policy_actual_normalized_action']!=pa['outgoing_raw_prior'],axis=1).sum()),
            physical_applied_prior_differs_raw_components=int((pa['policy_actual_normalized_action']!=pa['outgoing_raw_prior']).sum()),
            physical_native_clipped_rows=int(np.any(pa['policy_native_target_clipped'],axis=1).sum()),no_model_calls=True,no_native_steps=True).items():compare('context_alignment.'+name,alignment[name],value)
        old_frozen=read(bind(request['subjects']['source_frozen_inputs']['path']))
        compare('context_source_map',alignment['source_input_sha256'],old_frozen['input_sha256'])
        for key,path in request['context_paths'].items():
            name='source_output_manifest.json' if key=='manifest' else ('source_'+Path(path).name if key.startswith('schedule_') else Path(path).name)
            check('shared_byte_copy:'+key,sha(bind(shared/name))==sha(bind(path)))
        compare('context_output_map',alignment['context_sha256'],{name:sha(shared/(name+'.npy')) for name in ('nominal_context','center_context','physical_context')})
        ids=archive(shared/'data_identities.npz');expected_ids=dict(dataset=dataset,control=controls,phase=phases,source_frame=controls+11,center_to_nominal=np.arange(3057,dtype=np.int64),physical_successor=successor,physical_dataset=physical_dataset,physical_control=physical_control,axis_group=np.repeat(np.arange(6,dtype=np.int8),(3,3,23,3,3,23)),axis_radius=full_centers['axis_radius'])
        check('identity_keys',set(ids)==set(expected_ids))
        for name,value in expected_ids.items():exact('identity:'+name,ids[name],value)
        norm=archive(shared/'normalization.npz');cm,cv,cm32,cs32=moments(nominal_context)
        norm_schema={'feature_mean':((1323,),np.float32),'feature_std':((1323,),np.float32),
            'original_feature_mean':((1000,),np.float32),'original_feature_std':((1000,),np.float32),
            'context_mean':((323,),np.float32),'context_std':((323,),np.float32),'context_mean64':((323,),np.float64),'context_variance64':((323,),np.float64),
            'joint_span':((23,),np.float32),'default_q':((23,),np.float64),'joint_limits':((23,2),np.float64)}
        check('normalization_keys',set(norm)==set(norm_schema))
        for name,(shape,dtype) in norm_schema.items():check('norm_schema:'+name,norm[name].shape==shape and norm[name].dtype==dtype and np.isfinite(norm[name]).all())
        close('context_mean64',norm['context_mean64'],cm);close('context_variance64',norm['context_variance64'],cv)
        exact('context_mean_f32',norm['context_mean'],norm['context_mean64'].astype(np.float32));exact('context_std_f32',norm['context_std'],np.maximum(np.sqrt(norm['context_variance64']),.05).astype(np.float32))
        source_norm=archive(request['subjects']['normalization']['path'])
        for name in ('feature_mean','feature_std'):
            exact('reused_full_'+name,norm[name],source_norm[name]);exact('combined_original_'+name,norm[name][:1000],source_norm['original_'+name])
        exact('combined_context_mean',norm['feature_mean'][1000:],norm['context_mean']);exact('combined_context_std',norm['feature_std'][1000:],norm['context_std'])
        for name,value in [('joint_span',span),('default_q',default),('joint_limits',limits)]:exact('norm_'+name,norm[name],value)
        check('exact_normalized_blind_zero',np.count_nonzero((cm32-cm32)/cs32)==0)
        sc,sa,_,_=schedule(10000)
        for key,value in [('centers',sc),('axes',sa)]:
            old=npy(request['context_paths']['schedule_'+key]);check('old_schedule_shape:'+key,old.shape==(3000,864))
            exact('prior_schedule_prefix:'+key,old,value[:3000])
            exact('copied_prior_schedule:'+key,npy(shared/('source_schedule_'+key+'.npy')),old)
            exact('full_schedule_source:'+key,npy(request['schedule_paths']['schedule_'+key]),value)
            exact('shared_full_schedule:'+key,npy(shared/('schedule_'+key+'.npy')),value)
        compare('schedule_lineage',read(shared/'schedule_lineage.json'),dict(full_updates=10000,prior_updates=3000,prior_prefix_byte_exact=True,schedule_generated=False,source_centers_sha256=sha(request['schedule_paths']['schedule_centers']),source_axes_sha256=sha(request['schedule_paths']['schedule_axes']),prior_centers_sha256=sha(request['context_paths']['schedule_centers']),prior_axes_sha256=sha(request['context_paths']['schedule_axes'])))
        copied={('source_output_manifest.json' if key=='manifest' else ('source_'+Path(path).name if key.startswith('schedule_') else Path(path).name)):sha(path) for key,path in request['context_paths'].items()}
        compare('reused_inputs',read(shared/'reused_inputs.json'),dict(context_paths=request['context_paths'],schedule_paths=request['schedule_paths'],copied_sha256=copied,chronology_proof_sha256=request['subjects']['context_proof']['sha256'],context_reconstructed=False,normalization_recomputed=False,schedule_generated=False))
        data=dict(default=default,span=span,limits=limits,nominal_teacher=teacher,full_state_teacher=full_teacher,physical_teacher=phys('label_fixed_map_target'),physical_successor=successor,full_state_flags={name:full_array(name) for name in ('feedback_clipped','native_clipped')})
        runtime=read(shared/'runtime.json')
        check('deterministic_runtime',runtime['deterministic_algorithms'] is True and runtime['matmul_TF32'] is False and runtime['cudnn_TF32'] is False and runtime['AMP'] is False and runtime['CUBLAS_WORKSPACE_CONFIG']==':4096:8')
        stage='saved_condition_evidence'
        import torch
        import onnx
        torch.set_num_threads(1)
        check('saved_loader_versions',torch.__version__=='2.10.0+cu128' and onnx.__version__=='1.22.0')
        source=torch.load(bind(request['subjects']['checkpoint']['path']),map_location='cpu',weights_only=True)
        check('source71000',source['ordinary_final_step']==71000 and request['subjects']['checkpoint']['sha256']=='395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d')
        source_report=read(bind(request['subjects']['fit_report']['path']))
        check('qualified_source_report',source_report['completed'] is True and source_report['ordinary_final_step']==71000 and source_report['checkpoint_sha256']==request['subjects']['checkpoint']['sha256'])
        coefficient_path=bind(request['subjects']['coefficient_source']['path'])
        check('original_coefficient',read(coefficient_path)['coefficient']==source['full_state_coefficient']==WEIGHT)
        for name in ('feature_mean','feature_std'):exact('source_norm_'+name,source[name].numpy(),source_norm[name])
        energy=read(bind(request['subjects']['energy_source']['path']));group_weights=energy_weights(energy)
        for path,digest in energy['input_sha256'].items():bind(path,digest)
        calculated={};initializations={};qualification={};initial_outputs={};graph_reports={};report_hashes={}
        for condition in ('causal',):
            stage=condition;folder=fit
            report=read(bind(folder/'report.json'));report_hashes[condition]=sha(folder/'report.json')
            check(condition+'_optimization_and_diagnostics',report['optimization_completed'] is True and report['final_export_diagnostics_completed'] is True)
            required=['student_head.pt','student_head.onnx','initialization.pt','request.json','used_centers.npy','used_axes.npy','training_progress.npy','nominal_cell_losses.npy','full_state_cell_losses.npy','physical_cell_losses.npy','original_objectives.npy','balanced_full_state_cell_losses.npy','diagnostic_calls.jsonl','export_parity.json','drift.json','optimization_completed.json','restoration.json','promoted_parameters.json']
            required += [label+'_'+corpus+'.npy' for label in BACKENDS for corpus in CORPORA]+[label+'_metrics.json' for label in BACKENDS]
            required += ['drift_'+label+'_vs_final_GPU32_'+corpus+'.npy' for label in ('CPU64','GPU64','ORT64') for corpus in CORPORA]
            manifest(folder,folder/'output_manifest.json',required)
            for key,path in [('checkpoint_sha256',folder/'student_head.pt'),('onnx_sha256',folder/'student_head.onnx'),('normalization_sha256',shared/'normalization.npz'),('shared_manifest_sha256',shared/'output_manifest.json'),('training_request_sha256',req_path),('frozen_receipt_sha256',frozen_path),('output_manifest_sha256',folder/'output_manifest.json')]:check(condition+'_report_subject:'+key,report[key]==sha(path))
            for key,value in dict(condition=condition,ordinary_final_step=81000,additional_updates=10000,optimizer_step=16000,fresh_optimizer=False,features=1323,context_features=323,head_output='normalized_target',fixed_full_state_coefficient=WEIGHT,execution_dtype='float64',public_input_dtype='float32',public_output_dtype='float32',training_first_layer_execution='split_old256_new256_original1000_plus323',export_first_layer_execution='monolithic_float64_1323',parity_tolerance_rad=1e-5,all_frozen_inputs_unchanged=True,checkpoint_selection=False,native_steps=0,BFM_calls=0).items():compare(condition+'_report.'+key,report[key],value)
            for field,value in dict(ordinary_start_step=71000,optimizer_start_step=6000,hidden_width=512,architecture=[1323,512,512,23],expansion_seed=20260912,context_and_normalization_reused=True,source_checkpoint_sha256=request['subjects']['checkpoint']['sha256'],energy_source_sha256=request['subjects']['energy_source']['sha256']).items():compare('report_extra.'+field,report[field],value)
            compare(condition+'_carried_request',read(folder/'request.json'),dict(request,condition=condition,initialization_sha256=sha(folder/'initialization.pt'),shared_manifest_sha256=sha(shared/'output_manifest.json'),source_receipt_sha256=frozen_sha,clearance_sha256=sha(base/'training_clearance.json')))
            checkpoint=torch.load(bind(folder/'student_head.pt'),map_location='cpu',weights_only=True);init=torch.load(bind(folder/'initialization.pt'),map_location='cpu',weights_only=True);initializations[condition]=init
            names=['0.weight','0.bias','2.weight','2.bias','4.weight','4.bias'];shapes=((512,1323),(512,),(512,512),(512,),(23,512),(23,))
            check(condition+'_six_initial_tensors',set(init['actor_state'])==set(names))
            check('width_initial_actor_optimizer_RNG',not width_initial_errors(init,source))
            tree('retained_expansion_generator',checkpoint['expansion_generator_state'],init['expansion_generator_state'])
            for key,value in dict(ordinary_start_step=71000,optimizer_start_step=6000,hidden_width=512,expansion_seed=20260912,condition=condition,source_checkpoint_sha256=request['subjects']['checkpoint']['sha256']).items():compare(condition+'_init.'+key,init[key],value)
            tree(condition+'_final_rng',checkpoint['rng'],source['rng'])
            for name in ('feature_mean','feature_std'):exact(condition+'_init_'+name,init[name].numpy(),norm[name]);exact(condition+'_final_'+name,checkpoint[name].numpy(),norm[name])
            compare(condition+'_checkpoint_request',checkpoint['request'],request)
            for key,value in dict(kind='direct_absolute_native23_target',condition=condition,ordinary_final_step=81000,additional_updates=10000,optimizer_step=16000,fresh_optimizer=False,hidden_width=512,architecture=[1323,512,512,23],expansion_seed=20260912,context_order='previous_action23_then_incoming_history300',context_blinded=condition=='blinded',full_state_coefficient=WEIGHT,source_checkpoint_sha256=request['subjects']['checkpoint']['sha256']).items():compare(condition+'_checkpoint.'+key,checkpoint[key],value)
            for name,value in [('joint_span',span),('runtime_joint_span',span.astype(np.float64)),('default_q',default),('joint_limits',limits),('retained_feature_indices',kept)]:exact(condition+'_checkpoint_'+name,checkpoint[name].numpy(),value)
            check(condition+'_six_final_tensors',set(checkpoint['actor_state'])==set(names))
            for name,shape in zip(names,shapes):
                value=checkpoint['actor_state'][name];check(condition+'_finite:'+name,tuple(value.shape)==shape and value.dtype==torch.float32 and bool(torch.isfinite(value).all()))
            optimizer=checkpoint['optimizer_state'];check(condition+'_six_optimizer_states',set(optimizer['state'])==set(range(6)))
            for index,shape in enumerate(shapes):
                state=optimizer['state'][index];check(condition+'_optimizer_step:'+str(index),state['step'].dtype==torch.float32 and state['step'].shape==torch.Size([]) and float(state['step'])==16000)
                for key in ('exp_avg','exp_avg_sq'):check(condition+'_moment:'+str(index)+key,tuple(state[key].shape)==shape and state[key].dtype==torch.float32 and bool(torch.isfinite(state[key]).all()))
                check(condition+'_variance_nonnegative:'+str(index),bool((state['exp_avg_sq']>=0).all()))
            tree('final_whole_optimizer_group',optimizer['param_groups'],source['optimizer_state']['param_groups'])
            for label,opt,lr in [('initial',init['optimizer_state'],1e-6),('final',optimizer,1e-6)]:
                check(condition+'_'+label+'_one_group',len(opt['param_groups'])==1);g=opt['param_groups'][0]
                check(condition+'_'+label+'_optimizer_parameters',g['params']==list(range(6)) and g['lr']==lr and g['weight_decay']==1e-5 and g['foreach'] is False and g['fused'] is False and g['betas']==(0.9,0.999) and g['eps']==1e-8 and g['amsgrad'] is False)
            exact('checkpoint_group_weights',np.asarray(checkpoint['response_group_weights'],np.float64),group_weights)
            check('checkpoint_weight_rule',checkpoint['response_weight_rule']==PRODUCER_WEIGHT_RULE)
            optimization=read(folder/'optimization_completed.json')
            check(condition+'_optimization_receipt',optimization['optimization_completed'] is True and optimization['condition']==condition and optimization['ordinary_final_step']==81000 and optimization['optimizer_step']==16000 and optimization['checkpoint_sha256']==sha(folder/'student_head.pt') and optimization['export_validation_pending'] is True)
            exact(condition+'_used_centers',npy(folder/'used_centers.npy'),sc);exact(condition+'_used_axes',npy(folder/'used_axes.npy'),sa)
            progress=npy(folder/'training_progress.npy');check(condition+'_progress',progress.shape==(10000,6) and progress.dtype==np.float64 and np.isfinite(progress).all() and np.all(progress[:,:4]>=0) and np.all(progress[:,5]>=0))
            exact(condition+'_rates',progress[:,4],np.array([rate(i) for i in range(10000)],np.float64))
            cell_values={}
            for name,width in zip(CORPORA,(15,54,9)):
                cells=npy(folder/(name+'_cell_losses.npy'));check(condition+'_cell_shape:'+name,cells.shape==(10000,width) and cells.dtype==np.float64 and np.isfinite(cells).all() and np.all(cells>=0));cell_values[name]=cells
            weighted_cells=npy(folder/'balanced_full_state_cell_losses.npy')
            wanted=ledger_expectations(cell_values['nominal'],cell_values['full_state'],cell_values['physical'],group_weights)
            exact('balanced54_cells',weighted_cells,wanted['weighted_cells'])
            originals=npy(folder/'original_objectives.npy');check('original_objectives_schema',originals.shape==(10000,2) and originals.dtype==np.float64 and np.isfinite(originals).all())
            close('nominal_cells',progress[:,0],cell_values['nominal'].mean(axis=1),nominal=True)
            close('balanced_cells',progress[:,1],weighted_cells.mean(axis=1));close('physical_cells',progress[:,2],cell_values['physical'].mean(axis=1))
            close('original_full_cells',originals[:,0],cell_values['full_state'].mean(axis=1))
            close('balanced_total',progress[:,3],(progress[:,0]+WEIGHT*progress[:,1])+progress[:,2])
            close('original_total',originals[:,1],(progress[:,0]+WEIGHT*originals[:,0])+progress[:,2])
            outputs={label:{corpus:npy(folder/(label+'_'+corpus+'.npy')) for corpus in CORPORA} for label in BACKENDS}
            for label,values in outputs.items():
                for corpus,size in zip(CORPORA,SIZES):
                    a=values[corpus];check(condition+'_predictions:'+label+'/'+corpus,a.shape==(size,23) and a.dtype==np.float32 and np.isfinite(a).all())
            initial_outputs[condition]=outputs['initial_GPU32'];restoration=read(folder/'restoration.json');restored={corpus:drift(outputs['initial_GPU32'][corpus],npy(request['restoration_predictions'][corpus]),span) for corpus in CORPORA}
            maximum=max(v['max_preclip_error_rad'] for v in restored.values())
            compare(condition+'_restoration',restoration,dict(source_checkpoint_sha256=request['subjects']['checkpoint']['sha256'],**restoration_fields(),initial_drift=dict(passed=maximum<=1e-5,tolerance_rad=1e-5,max_preclip_error_rad=maximum,corpora=restored,source_ordinary_step=71000,changed_MatMul_dimension=True,stored_first_layer_width=1323,stored_hidden_width=512,old_block_contractions_preserved=True,first_layer_execution='split_old256_new256_original1000_plus323',byte_gate_required=False)))
            check(condition+'_initial_drift_gate',maximum<=1e-5)
            compare('report_initial_restoration',report['initial_restoration'],restoration['initial_drift'])
            compare('report_group_weights',report['response_group_weights'],group_weights.tolist())
            check('report_weight_rule',report['response_weight_rule']==PRODUCER_WEIGHT_RULE)
            compare('report_direct_subjects',report['direct_subject_sha256'],{key:subject['sha256'] for key,subject in request['subjects'].items()})
            compare('training_columns',report['training_loss_columns'],['nominal','balanced_full_state','physical','balanced_total','learning_rate','preclip_gradient_norm'])
            calculated[condition]={}
            for label in BACKENDS:
                m=balanced_metrics(metrics(outputs[label],data),group_weights,energy);calculated[condition][label]=m;saved=read(folder/(label+'_metrics.json'));compare(condition+'_metrics:'+label,saved,m);compare(condition+'_report_metrics:'+label,report['metrics'][label],saved)
            comparisons=numerical_comparisons(outputs,data);maximum=max(comparisons.values());qualification[condition]=maximum<=1e-5
            parity=read(folder/'export_parity.json');compare(condition+'_nine_parity',parity['comparisons'],comparisons)
            check(condition+'_parity_flags',parity['maximum_preclamp_rad']==maximum and parity['tolerance_rad']==1e-5 and parity['nonfinite_comparisons']==[] and parity['passed'] is qualification[condition] and report['max_preclip_error_rad']==maximum)
            for key in ('completed','numerical_gate_passed','export_parity_passed'):check(condition+'_release:'+key,report[key] is qualification[condition])
            drift_report=read(folder/'drift.json');check(condition+'_nine_drifts',len(drift_report)==9)
            for label in ('CPU64','GPU64','ORT64'):
                for corpus in CORPORA:
                    key=label+'_vs_final_GPU32_'+corpus;array,summary=drift_summary(outputs['final_GPU32'][corpus],outputs[label][corpus],data);exact(condition+'_drift:'+key,npy(folder/('drift_'+key+'.npy')),array);compare(condition+'_drift_summary:'+key,drift_report[key],summary)
            graph=onnx.load(bind(folder/'student_head.onnx'));onnx.checker.check_model(graph);graph_reports[condition]=audit_graph(graph,checkpoint,norm)
            promoted=read(folder/'promoted_parameters.json')
            expected_promoted={name:dict(shape=list(shape),dtype='torch.float64',source_float32_roundtrip_exact=True) for name,shape in [('feature_mean',(1323,)),('feature_std',(1323,)),('w0',(512,1323)),('b0',(512,)),('w1',(512,512)),('b1',(512,)),('w2',(23,512)),('b2',(23,))]}
            compare(condition+'_promoted_parameters',promoted,dict(parameters=expected_promoted,checkpoint_sha256=sha(folder/'student_head.pt')))
            count=report['counts'];compare(condition+'_training_count',count['training'],counts(30000,146860000))
            for label in BACKENDS:compare(condition+'_diagnostic_count:'+label,count['diagnostics'][label],counts(1437,367570))
            for key in ('calibration_forward_calls','calibration_gradient_calls','BFM_calls','native_calls','manual_export_trace_calls'):check(condition+'_zero:'+key,count[key]==0)
            compare(condition+'_checkpoint_training',checkpoint['counters']['training'],count['training'])
            for label in BACKENDS:compare(condition+'_checkpoint_diagnostics:'+label,checkpoint['counters']['diagnostics'][label],counts(1437,367570) if label=='initial_GPU32' else counts(0,0))
            ledger=[json.loads(line) for line in (folder/'diagnostic_calls.jsonl').read_text(encoding='utf-8').splitlines()]
            expected=[dict(backend=label,corpus=corpus,start=start,stop=min(start+256,size),returned=True,synchronized=True,verified=True) for label in BACKENDS for corpus,size in zip(CORPORA,SIZES) for start in range(0,size,256)]
            check(condition+'_7185_call_partitions',ledger==expected and len(ledger)==7185)
        qualified=qualification['causal']
        final=calculated['causal']['ORT64']
        stage='process_accounting'
        start=read(bind(base/'fit_process_v1/start.json'));child=read(bind(base/'fit_process_v1/child.json'));exit_record=read(bind(base/'fit_process_v1/exit.json'))
        check('pid_links',start['wrapper_pid']==child['wrapper_pid']==exit_record['wrapper_pid'] and child['child_pid']==exit_record['child_pid'] and child['captured_handle_nonzero'] is True)
        check('start_identities',start['request_sha256']==training_sha and start['frozen_receipt_sha256']==frozen_sha and start['clearance_sha256']==sha(base/'training_clearance.json'))
        check('known_consistent_exit',exit_record['exit_known'] is True and exit_record['child_started'] is True and exit_record['all_postrun_pins_exact'] is True and exit_record['raw_python_exit_code']==exit_record['exit_code']==(0 if qualified else 1) and (not qualified or exit_record['error'] is None))
        check('exit_clearance',exit_record['clearance_sha256']==sha(base/'training_clearance.json'))
        required_pins=dict(frozen['input_sha256']);required_pins.update({str(Path(frozen['source_directory'])/n):d for n,d in frozen['source_sha256'].items()});required_pins.update({str(req_path):training_sha,str(frozen_path):frozen_sha,str(base/'training_clearance.json'):sha(base/'training_clearance.json'),clearance['review_path']:clearance['review_sha256']})
        expected_pins={canonical(p):d for p,d in required_pins.items()}
        for filename in ('prerun_pins.json','postrun_pins.json'):
            p=read(bind(base/'fit_process_v1'/filename));check(filename+'_all_exact',p['all_exact'] is True);actual={canonical(k):v for k,v in p['files'].items()};check(filename+'_membership',set(actual)==set(expected_pins))
            check(filename+'_count',p['count']==len(expected_pins) and len(actual)==len(p['files']))
            for path,digest in expected_pins.items():check(filename+':'+path,actual[path]['expected']==actual[path]['actual']==digest and actual[path]['matched'] is True)
        if qualified:
            check('no_failed_stage',not (fit/'failure.json').exists() and not (fit/'setup_failure.json').exists())
        else:
            failure=read(bind(fit/'failure.json'))
            check('preserved_numerical_failure',failure['optimization_completed'] is True and failure['final_export_diagnostics_completed'] is True and failure['automatic_retry'] is False and failure['ordinary_endpoint_preserved'] is True and failure['all_frozen_inputs_unchanged'] is True and failure['preservation_errors']==[] and failure['error']=="ValueError('Endpoint same-weight FP64 parity failed; continuation incomplete.')")
        role_paths=release_paths(base,request)
        for role,path in role_paths.items():literal(ar['subjects'][role],path,role)
        owner_path=base/'owner_completion_verification.json';literal(ar['subjects']['owner_completion'],owner_path,'owner_completion');owner=read(owner_path)
        for name in ('owner_verification_passed','accounting_passed','processes_absent','raw_exit_known','all_postrun_pins_exact'):check('owner_flag:'+name,owner[name] is True)
        check('owner_qualification',owner['completion_passed'] is qualified and owner['numerical_completion_passed'] is qualified and owner['optimization_completed'] is True)
        check('owner_exit_and_pids',owner['raw_python_exit_code']==owner['exit_code']==exit_record['exit_code'] and owner['expected_pids']==[exit_record['wrapper_pid'],exit_record['child_pid']])
        current_pins={canonical(p):d for p,d in frozen['input_sha256'].items()};current_pins.update({canonical(Path(frozen['source_directory'])/n):d for n,d in frozen['source_sha256'].items()})
        check('owner_counts',owner['launch_pin_count']==len(expected_pins) and owner['current_pin_count']==len(current_pins) and owner['condition']=='causal')
        owner_expected={role:sha(path) for role,path in role_paths.items()}
        owner_expected.update(process_exit=sha(base/'fit_process_v1/exit.json'),source_fit_report=request['subjects']['fit_report']['sha256'],balance_source_review=request['subjects']['balance_source_review']['sha256'])
        if not qualified:owner_expected['failure']=sha(fit/'failure.json')
        compare('owner_direct_subjects',owner['direct_subject_sha256'],owner_expected)
        stage='final_rehash'
        for path,digest in list(tracked.items()):check('final_hash:'+path,sha(path)==digest)
        result=dict(passed=qualified,evidence_audit_passed=True,optimization_complete=True,export_qualified=qualified,
            canonical_evaluation_cleared=False,checks=len(checks),training_request_sha256=training_sha,fit_report_sha256=sha(fit/'report.json'),
            audit_request_sha256=request_sha,metrics=calculated,graph=graph_reports,final_metrics=final,
            direct_subject_sha256={role:sha(path) for role,path in role_paths.items()},input_sha256=tracked,
            model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,
            limitations=['Saved predictions and training cells are checked for algebra and provenance without recomputing model authenticity or optimizer gradients.',
                'Initialization checks reconstruct saved tensors with a local CPU generator; they do not execute a model. New moment entries retain shared age6000.',
                'Nominal float32 reductions use3e-7 relative tolerance; other saved metric reductions5e-12. Initial dimensional drift retains the original1e-5rad gate.',
                'Process absence is established separately by the bound completion owner. No context-utility result proves closed-loop stability or uniquely identifies hidden-state causation.'])
    except BaseException as error:
        result=dict(passed=False,evidence_audit_passed=False,optimization_complete=False,export_qualified=False,canonical_evaluation_cleared=False,
            stage=stage,error=repr(error),checks=len(checks),input_sha256=tracked,model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0)
        raise
    finally:
        write(output/'checks.json',checks);write(output/'report.json',result)
    print(json.dumps(dict(evidence_audit_passed=result['evidence_audit_passed'],checks=len(checks),report_sha256=sha(output/'report.json'))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True);p.add_argument('--request-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.request,a.request_sha256,a.output)
