"""Audit completed full-state fit from saved evidence only, including numerical failure.

No model construction, forwards, gradients, optimizer steps, ORT or native calls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_math import phase_indices
from audit_full_state_math import (BACKENDS,CORPORA,SIZES,AXES,schedule,rate,gradient_geometry,
    initial_losses,metrics,numerical_comparisons,drift_summary)
from audit_restoration import differences
from audit_graph import audit_graph

def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('w',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def normalized_path(path):return Path(path).resolve().as_posix()

def run(request_path,request_sha,output):
    output.mkdir(exist_ok=False)
    checked=[];tracked={};stage='request';result=None
    def check(name,condition):
        checked.append(dict(name=name,passed=bool(condition)))
        if not condition:raise AssertionError(name)
    def bind(path,digest=None):
        path=Path(path);key=normalized_path(path);actual=sha(path)
        if digest is not None:check('hash:'+key,actual==digest)
        if key in tracked:check('stable:'+key,tracked[key]==actual)
        tracked[key]=actual;return path
    def exact(name,a,b):
        a,b=np.asarray(a),np.asarray(b)
        check(name,a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes())
    def close(name,a,b,nominal=False):
        a,b=np.asarray(a),np.asarray(b)
        check(name,a.shape==b.shape and np.allclose(a,b,rtol=3e-7 if nominal else 5e-12,atol=1e-14,equal_nan=False))
    def compare(name,actual,expected):
        if isinstance(expected,dict):
            check(name+'.keys',isinstance(actual,dict) and set(actual)==set(expected))
            for k,v in expected.items():compare(name+'.'+k,actual[k],v)
        elif isinstance(expected,list):
            check(name+'.length',isinstance(actual,list) and len(actual)==len(expected))
            for i,v in enumerate(expected):compare(name+'.'+str(i),actual[i],v)
        elif isinstance(expected,float):close(name,actual,expected,nominal=name.endswith('normalized_MSE') or name.endswith('nominal_objective') or name.endswith('weighted_objective'))
        else:check(name,type(actual)==type(expected) and actual==expected)
    def npy(path):return np.load(bind(path),mmap_mode='r',allow_pickle=False)
    def archive(path):
        with np.load(bind(path),allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
    def tree(name,a,b):check(name,not differences(a,b,name))
    def counts(calls,rows):return dict(calls_attempted=calls,calls_returned=calls,calls_synchronized=calls,calls_verified=calls,rows_attempted=rows,rows_returned=rows,rows_verified=rows)
    try:
        bind(request_path,request_sha);audit_request=read(request_path)
        check('audit_kind',audit_request['kind']=='saved_full58_fit_evidence_only')
        import sys
        check('audit_runtime',Path(sys.executable).resolve()==Path(audit_request['python_path']).resolve())
        check('audit_numpy_version',np.__version__=='1.23.5')
        for path,digest in audit_request['source_sha256'].items():bind(path,digest)
        review=read(bind(audit_request['source_review']['path'],audit_request['source_review']['sha256']))
        check('audit_source_clear',review[audit_request['source_review']['pass_field']] is True)
        for path,digest in audit_request['source_sha256'].items():check('reviewed_audit_source:'+Path(path).name,review['source_sha256'][Path(path).name]==digest)
        for role,subject in audit_request['subjects'].items():bind(subject['path'],subject['sha256'])
        experiment=Path(audit_request['experiment']);fit=experiment/'fit'
        request=read(bind(experiment/'training_request.json'));frozen=read(bind(experiment/'training_frozen_inputs.json'))
        check('known_training_request',sha(experiment/'training_request.json')=='0c3bbc4e8567f6dfc09dfe7d9cdfcdd8af5a56178ea112af3a647d84c48d0dea')
        check('known_frozen_receipt',sha(experiment/'training_frozen_inputs.json')=='bbb67b48187dcead0dc0886496573447ffdd58582eaeb63f41a7fa947e922f18')
        check('request_receipt_binding',frozen['training_request_sha256']==sha(experiment/'training_request.json'))
        for path,digest in frozen['input_sha256'].items():bind(path,digest)
        for name,digest in frozen['source_sha256'].items():bind(Path(frozen['source_directory'])/name,digest)
        clearance=read(bind(experiment/'training_clearance.json'))
        check('clearance',clearance['approved'] is True and clearance['request_sha256']==sha(experiment/'training_request.json') and clearance['frozen_receipt_sha256']==sha(experiment/'training_frozen_inputs.json'))
        launch_review=read(bind(clearance['review_path'],clearance['review_sha256']));bind(clearance['launcher_path'],clearance['launcher_sha256'])
        check('launch_review',launch_review[clearance['review_pass_field']] is True)
        for key,path in [('training_request',experiment/'training_request.json'),('frozen_inputs',experiment/'training_frozen_inputs.json'),('launcher',Path(clearance['launcher_path']))]:
            subject=launch_review['subjects'][key];check('literal_launch_subject:'+key,Path(subject['path']).resolve()==path.resolve() and subject['sha256']==sha(path))
        report=read(bind(fit/'report.json'));out_manifest=read(bind(fit/'output_manifest.json'))
        check('completed_optimization_and_diagnostics',report['optimization_completed'] is True and report['final_export_diagnostics_completed'] is True)
        for name,value in dict(ordinary_final_step=65000,additional_updates=10000,optimizer_step=10000,restored_start_step=55000,fresh_optimizer=True,
            features=1000,head_output='normalized_target',nominal_rows=9904,full_state_endpoint_rows=354612,physical_rows=3054,full_state_cells=54,
            execution_dtype='float64',public_input_dtype='float32',public_output_dtype='float32',ELU_implementation='Where(x>0,x,Exp(Min(x,0))-1)',
            parity_tolerance_rad=1e-5,all_frozen_inputs_unchanged=True,checkpoint_selection=False,FP32_ONNX_release=False,BFM_calls=0,native_steps=0,hardware_authorized=False).items():check('report.'+name,type(report[name])==type(value) and report[name]==value)
        check('output_manifest_subject',report['output_manifest_sha256']==sha(fit/'output_manifest.json'))
        check('output_manifest_request',out_manifest['training_request_sha256']==sha(experiment/'training_request.json') and out_manifest['frozen_receipt_sha256']==sha(experiment/'training_frozen_inputs.json'))
        for name,digest in out_manifest['files'].items():
            path=(fit/name).resolve();check('safe_output:'+name,path.parent==fit.resolve());bind(path,digest)
        required=['student_head.pt','student_head.onnx','normalization.npz','initialization.pt','request.json','data_identities.npz','nominal_normalized_labels.npy',
            'schedule_centers.npy','schedule_axes.npy','used_centers.npy','used_axes.npy','sampler.json','coefficient.json','calibration_forward_outputs.npz','calibration_gradients.npz',
            'training_progress.npy','nominal_cell_losses.npy','full_state_cell_losses.npy','physical_cell_losses.npy','diagnostic_calls.jsonl','export_parity.json','drift.json',
            'optimization_completed.json','restoration.json','promoted_parameters.json']
        required += [label+'_'+corpus+'.npy' for label in BACKENDS for corpus in CORPORA]+[label+'_metrics.json' for label in BACKENDS]
        required += ['drift_'+label+'_vs_final_GPU32_'+corpus+'.npy' for label in ('CPU64','GPU64','ORT64') for corpus in CORPORA]
        check('all_required_outputs_manifest_bound',set(required)<=set(out_manifest['files']))
        for key,path in [('training_request_sha256',experiment/'training_request.json'),('frozen_receipt_sha256',experiment/'training_frozen_inputs.json'),
            ('checkpoint_sha256',fit/'student_head.pt'),('onnx_sha256',fit/'student_head.onnx'),('normalization_sha256',fit/'normalization.npz'),
            ('coefficient_sha256',fit/'coefficient.json'),('restoration_sha256',fit/'restoration.json')]:check('report_subject:'+key,report[key]==sha(path))
        carried=read(fit/'request.json');compare('carried_request',carried,dict(**request,source_receipt_sha256=sha(experiment/'training_frozen_inputs.json'),
            clearance_sha256=sha(experiment/'training_clearance.json'),initialization_sha256=sha(fit/'initialization.pt'),normalization_sha256=sha(fit/'normalization.npz'),sampler_sha256=sha(fit/'sampler.json')))
        stage='data_identity'
        paths=request['paths'];centers=archive(paths['centers']);pico=archive(paths['pico']);walk=archive(paths['walk002'])
        kept=np.r_[0:52,75:1023].astype(np.int64)
        features=np.concatenate([a['features'][:,kept] for a in (centers,pico,walk)])
        teacher=np.concatenate([a['expert_target'] for a in (centers,pico,walk)])
        check('nominal_schema',features.shape==(9904,1000) and features.dtype==np.float32 and teacher.shape==(9904,23) and teacher.dtype==np.float64)
        dataset=np.r_[np.repeat(np.arange(3,dtype=np.int64),1019),np.full(5980,3,np.int64),np.full(867,4,np.int64)]
        control=np.r_[np.tile(np.arange(250,1269,dtype=np.int64),3),np.arange(250,6230,dtype=np.int64),np.arange(250,1117,dtype=np.int64)]
        phase=np.concatenate([np.full(len(ids),i%3,np.int64) for i,ids in enumerate(phase_indices())])
        exact('source_dataset',centers['dataset'],dataset[:3057]);exact('source_center_controls',centers['control'],control[:3057])
        for name,a,start,stop in [('centers',centers,0,3057),('pico',pico,3057,9037),('walk002',walk,9037,9904)]:
            exact(name+'_control',a['control'],control[start:stop]);exact(name+'_frame',a['source_frame'],control[start:stop]+11)
            if name!='centers':exact(name+'_phase',a['phase'],phase[start:stop])
        contract=read(bind(paths['contract']));default=np.array(contract['default_q'],np.float64);limits=np.array(contract['joint_limits'],np.float64);span=centers['joint_span']
        exact('native_span_f32',span,np.diff(limits,axis=1).ravel().astype(np.float32))
        exact('nominal_labels',npy(fit/'nominal_normalized_labels.npy'),((teacher-default)/span.astype(np.float64)).astype(np.float32))
        full_manifest=read(bind(request['full_state_paths']['manifest']));full_root=Path(request['full_state_paths']['manifest']).parent
        def full_array(name):
            spec=full_manifest['arrays'][name];path=full_root/spec['path'];bind(path,spec['sha256']);a=npy(path)
            check('full_schema:'+name,list(a.shape)==spec['shape'] and str(a.dtype)==spec['dtype']);return a
        full_centers=archive(full_root/'centers.npz')
        exact('full_centers_features',full_centers['features'],features[:3057]);exact('full_centers_targets',full_centers['target'],teacher[:3057])
        for name,a in [('dataset',dataset[:3057]),('control',control[:3057]),('source_frame',control[:3057]+11),('phase',phase[:3057].astype(np.int8))]:exact('full_center_'+name,full_centers[name],a)
        full_teacher=full_array('target').reshape(354612,23);full_change=full_array('target_change').reshape(354612,23)
        for start in range(0,354612,8192):
            end=min(start+8192,354612);indices=np.arange(start,end)//116
            exact('full_target_change:'+str(start),full_change[start:end],full_teacher[start:end]-teacher[indices])
        check('full_status_all_valid',bool(np.all(full_array('status')==1)))
        physical_manifest=read(bind(paths['physical_manifest']));physical_root=Path(paths['physical_manifest']).parent
        def phys(name):
            spec=physical_manifest['arrays'][name];path=physical_root/spec['path'];bind(path,spec['sha256']);return npy(path)
        physical_dataset=np.repeat(np.arange(3,dtype=np.int64),1018);physical_control=np.tile(np.arange(251,1269,dtype=np.int64),3)
        successor=physical_dataset*1019+physical_control-250
        for name,a in [('dataset',physical_dataset),('successor_control',physical_control),('successor_center_index',successor)]:exact('physical_'+name,phys(name),a)
        check('physical_valid',bool(np.all(phys('label_valid'))))
        data=dict(default=default,span=span,limits=limits,nominal_teacher=teacher,full_state_teacher=full_teacher,
            physical_teacher=phys('label_fixed_map_target'),physical_successor=successor,
            full_state_flags={name:full_array(name) for name in ('feedback_clipped','native_clipped')})
        ids=archive(fit/'data_identities.npz')
        values=dict(dataset=dataset,control=control,phase=phase,source_frame=control+11,center_to_nominal=np.arange(3057,dtype=np.int64),
            physical_successor=successor,physical_dataset=physical_dataset,physical_control=physical_control,
            full_state_dataset=dataset[:3057],full_state_control=control[:3057],full_state_phase=phase[:3057].astype(np.int8),full_state_cell=(dataset[:3057]*3+phase[:3057]).astype(np.int8),
            axis_group=np.repeat(np.arange(6,dtype=np.int8),(3,3,23,3,3,23)),axis_radius=full_centers['axis_radius'],axis_units=full_centers['axis_units'])
        check('identity_keys',set(ids)==set(values))
        for key,value in values.items():exact('identity:'+key,ids[key],value)
        for name in ('full_state_root_audit','full_state_data_review'):
            subject=request['subjects'][name];q=read(bind(subject['path'],subject['sha256']));check(name,q[subject['pass_field']] is True)
            for key,value in subject['required_fields'].items():check(name+'.'+key,q[key]==value)
        data_owner_subject=request['subjects']['full_state_root_owner'];data_owner=read(bind(data_owner_subject['path'],data_owner_subject['sha256']))
        check('corrected_data_owner_v2',data_owner_subject['sha256']=='3f47a899d3e08ebb9d5a1ec514e395563dd677c4209a281f774f76401b401e43' and data_owner['passed'] is True)
        stage='sampler_and_calibration'
        sample_c,sample_a,before,after=schedule()
        exact('all_sampled_centers',npy(fit/'schedule_centers.npy'),sample_c);exact('all_sampled_axes',npy(fit/'schedule_axes.npy'),sample_a)
        exact('all_executed_centers',npy(fit/'used_centers.npy'),sample_c);exact('all_executed_axes',npy(fit/'used_axes.npy'),sample_a)
        sampler=read(fit/'sampler.json');compare('sampler_initial',sampler['initial_state'],before);compare('sampler_final',sampler['final_state'],after)
        check('sampler_runtime_draws',sampler['runtime_draws']==0 and sampler['replacement'] is True and sampler['seed']==20260911)
        check('sampler_bound_arrays',sampler['schedule_centers_sha256']==sha(fit/'schedule_centers.npy') and sampler['schedule_axes_sha256']==sha(fit/'schedule_axes.npy'))
        geometry=gradient_geometry(archive(fit/'calibration_gradients.npz'));coefficient=read(fit/'coefficient.json')
        for key,value in geometry.items():compare('gradient_geometry.'+key,coefficient[key],value)
        weight=geometry['coefficient'];check('frozen_coefficient',report['full_state_coefficient']==coefficient['coefficient']==weight)
        check('calibration_subjects',coefficient['gradient_sha256']==sha(fit/'calibration_gradients.npz') and coefficient['forward_outputs_sha256']==sha(fit/'calibration_forward_outputs.npz'))
        calibration=archive(fit/'calibration_forward_outputs.npz');losses,cells=initial_losses(calibration,data,sample_c[0],sample_a[0])
        for name in CORPORA:
            close('calibration_loss.'+name,coefficient['losses'][name],losses[name],nominal=name=='nominal')
            close('calibration_cells.'+name,calibration[name+'_cell_losses'],cells[name],nominal=name=='nominal')
        check('calibration_flags',coefficient['schedule_index']==0 and coefficient['optimizer_updates']==0 and coefficient['model_unchanged'] is True and coefficient['RNG_unchanged'] is True)
        compare('calibration_gradient_count',coefficient['counters'],dict(attempted=3,returned=3,synchronized=3,verified=3))
        check('calibration_parameter_order',coefficient['parameter_names']==['0.weight','0.bias','2.weight','2.bias','4.weight','4.bias'])
        progress=npy(fit/'training_progress.npy');check('complete_training_log',progress.shape==(10000,6) and progress.dtype==np.float64 and np.isfinite(progress).all())
        check('nonnegative_losses_and_gradnorm',bool(np.all(progress[:,:4]>=0) and np.all(progress[:,5]>=0)))
        exact('all_learning_rates',progress[:,4],np.array([rate(i) for i in range(10000)],np.float64))
        saved_cells={name:npy(fit/(name+'_cell_losses.npy')) for name in CORPORA}
        for name,width in [('nominal',15),('full_state',54),('physical',9)]:check('loss_cell_schema:'+name,saved_cells[name].shape==(10000,width) and saved_cells[name].dtype==np.float64 and np.isfinite(saved_cells[name]).all() and np.all(saved_cells[name]>=0))
        for i,name in enumerate(CORPORA):close('cell_mean:'+name,progress[:,i],saved_cells[name].mean(axis=1),nominal=name=='nominal')
        close('fixed_weight_loss_composition',progress[:,3],(progress[:,0]+weight*progress[:,1])+progress[:,2])
        stage='restoration_and_checkpoint'
        import torch
        import onnx
        check('audit_saved_loader_versions',torch.__version__=='2.10.0+cu128' and onnx.__version__=='1.22.0')
        checkpoint=torch.load(bind(fit/'student_head.pt'),map_location='cpu',weights_only=True)
        source_subject=request['subjects']['checkpoint'];source=torch.load(bind(source_subject['path'],source_subject['sha256']),map_location='cpu',weights_only=True)
        init=torch.load(bind(fit/'initialization.pt'),map_location='cpu',weights_only=True)
        check('source_55000',source_subject['sha256']=='9ceef5099ebd154e08a1c2c3784c4e06021d464e607474548763665b24f8f9e7' and source['ordinary_final_step']==55000)
        check('initialization_flags',init['ordinary_start_step']==55000 and init['optimizer_start_step']==0 and init['fresh_optimizer'] is True and init['fresh_actor_initialization'] is False and init['source_checkpoint_sha256']==source_subject['sha256'])
        tree('exact_initial_actor',init['actor_state'],source['actor_state']);tree('exact_initial_rng',init['rng_after_restoration'],source['rng']);tree('exact_final_rng',checkpoint['rng'],source['rng'])
        check('fresh_initial_optimizer',init['optimizer_state']['state']=={})
        for name in ('feature_mean','feature_std'):tree('initial_'+name,init[name],source[name])
        norm=archive(fit/'normalization.npz');check('normalization_archive_byte_identity',sha(fit/'normalization.npz')==request['subjects']['normalization']['sha256'])
        for name,value in dict(joint_span=span,runtime_joint_span=span.astype(np.float64),default_q=default,joint_limits=limits,retained_feature_indices=kept).items():exact('normalization_'+name,norm[name],value)
        for name in ('feature_mean','feature_std','joint_span','runtime_joint_span','default_q','joint_limits','retained_feature_indices'):
            exact('final_checkpoint_'+name,checkpoint[name].numpy(),norm[name]);tree('checkpoint_source_'+name,checkpoint[name],source[name])
        compare('checkpoint_request',checkpoint['request'],request)
        for key,value in dict(kind='direct_absolute_native23_target',ordinary_final_step=65000,additional_updates=10000,optimizer_step=10000,fresh_optimizer=True,
            seed=20260911,output='normalized_absolute_target',full_state_coefficient=weight,coefficient_sha256=sha(fit/'coefficient.json'),source_checkpoint_sha256=source_subject['sha256']).items():check('checkpoint.'+key,checkpoint[key]==value)
        shapes=((256,1000),(256,),(256,256),(256,),(23,256),(23,));names=['0.weight','0.bias','2.weight','2.bias','4.weight','4.bias']
        check('six_actor_tensors',set(checkpoint['actor_state'])==set(names))
        for name,shape in zip(names,shapes):
            a=checkpoint['actor_state'][name];check('actor_finite:'+name,a.dtype==torch.float32 and tuple(a.shape)==shape and bool(torch.isfinite(a).all()))
        optimizer=checkpoint['optimizer_state'];check('six_optimizer_states',len(optimizer['state'])==6)
        for order,state in enumerate(optimizer['state'].values()):
            check('optimizer_step:'+str(order),float(state['step'])==10000)
            for key in ('exp_avg','exp_avg_sq'):check('optimizer_moment:'+str(order)+key,tuple(state[key].shape)==shapes[order] and state[key].dtype==torch.float32 and bool(torch.isfinite(state[key]).all()))
        for label,opt,lr in [('initial',init['optimizer_state'],1e-4),('final',optimizer,1e-5)]:
            check(label+'_one_optimizer_group',len(opt['param_groups'])==1);g=opt['param_groups'][0]
            check(label+'_optimizer_hyperparameters',g['lr']==lr and g['weight_decay']==1e-5 and g['foreach'] is False and g['fused'] is False and g['betas']==(0.9,0.999) and g['eps']==1e-8 and not g['amsgrad'] and len(g['params'])==6)
        optimization=read(fit/'optimization_completed.json');check('optimization_subject',optimization['optimization_completed'] is True and optimization['ordinary_final_step']==65000 and optimization['optimizer_step']==10000 and optimization['checkpoint_sha256']==sha(fit/'student_head.pt') and optimization['coefficient_sha256']==sha(fit/'coefficient.json') and optimization['export_validation_pending'] is True)
        stage='complete_saved_predictions'
        outputs={label:{corpus:npy(fit/(label+'_'+corpus+'.npy')) for corpus in CORPORA} for label in BACKENDS}
        for label,values in outputs.items():
            for corpus,size in zip(CORPORA,SIZES):
                a=values[corpus];check('predictions:'+label+'/'+corpus,a.shape==(size,23) and a.dtype==np.float32 and np.isfinite(a).all())
        for corpus in ('nominal','physical'):exact('initial_old_gpu_partition:'+corpus,outputs['initial_GPU32'][corpus],npy(request['restoration_predictions'][corpus]))
        overlap=outputs['initial_GPU32']['full_state'].reshape(3057,58,2,23)[:,35:].reshape(140622,23)
        delta=(overlap.astype(np.float64)-npy(request['restoration_predictions']['velocity']).astype(np.float64))*span.astype(np.float64)
        restoration=read(fit/'restoration.json')
        check('restoration_pass',restoration['all_restoration_checks_passed'] is True and restoration['fresh_optimizer'] is True and restoration['optimizer_step']==0 and restoration['optimizer_state_count']==0 and restoration['source_checkpoint_sha256']==source_subject['sha256'])
        check('overlap_drift_scope',restoration['overlap_partition_changed'] is True and restoration['overlap_byte_gate_required'] is False)
        close('overlap_drift_max',restoration['overlap_max_preclip_difference_rad'],float(np.abs(delta).max()));close('overlap_drift_RMS',restoration['overlap_RMS_preclip_difference_rad'],float(np.sqrt(np.mean(delta**2))))
        calculated={}
        for label in BACKENDS:
            m=metrics(outputs[label],data);m['weighted_objective']=m['nominal_objective']+weight*m['full_state_objective']+m['physical_objective'];calculated[label]=m
            saved_metrics=read(fit/(label+'_metrics.json'));compare('metrics:'+label,saved_metrics,m);compare('report_metrics:'+label,report['metrics'][label],saved_metrics)
        comparisons=numerical_comparisons(outputs,data);maximum=max(comparisons.values());qualified=maximum<=1e-5
        parity=read(fit/'export_parity.json');compare('nine_parity_values',parity['comparisons'],comparisons)
        check('numerical_identity',parity['maximum_preclamp_rad']==maximum and parity['tolerance_rad']==1e-5 and parity['nonfinite_comparisons']==[] and parity['passed'] is qualified)
        check('numerical_report_max',report['max_preclip_error_rad']==maximum);compare('report_parity',report['export_parity'],parity)
        for key in ('completed','numerical_gate_passed','export_parity_passed'):check('actual_release_flag:'+key,report[key] is qualified)
        drifts=read(fit/'drift.json')
        check('nine_drift_entries',len(drifts)==9)
        for label in ('CPU64','GPU64','ORT64'):
            for corpus in CORPORA:
                key=label+'_vs_final_GPU32_'+corpus;d,summary=drift_summary(outputs['final_GPU32'][corpus],outputs[label][corpus],data)
                exact('drift_array:'+key,npy(fit/('drift_'+key+'.npy')),d);compare('drift_summary:'+key,drifts[key],summary)
        stage='graph_and_call_accounting'
        graph=onnx.load(bind(fit/'student_head.onnx'));onnx.checker.check_model(graph);graph_report=audit_graph(graph,checkpoint,norm)
        count=report['counters'];compare('calibration_count',count['calibration'],counts(3,14686));compare('training_count',count['training'],counts(30000,146860000))
        compare('gradient_count',count['gradients'],dict(attempted=3,returned=3,synchronized=3,verified=3))
        for label in BACKENDS:compare('backend_count:'+label,count['diagnostics'][label],counts(1437,367570))
        for name in ('native_calls','BFM_calls','manual_export_trace_calls'):check('zero_'+name,count[name]==0)
        for name in ('calibration','training','gradients'):compare('checkpoint_counter:'+name,checkpoint['counters'][name],count[name])
        compare('checkpoint_initial_diagnostics',checkpoint['counters']['diagnostics']['initial_GPU32'],counts(1437,367570))
        for name in BACKENDS[1:]:compare('checkpoint_pending_backend:'+name,checkpoint['counters']['diagnostics'][name],counts(0,0))
        ledger=[json.loads(line) for line in (fit/'diagnostic_calls.jsonl').read_text(encoding='utf-8').splitlines()]
        expected=[]
        for label in BACKENDS:
            for corpus,size in zip(CORPORA,SIZES):
                for start in range(0,size,256):expected.append(dict(backend=label,corpus=corpus,start=start,stop=min(start+256,size),returned=True,synchronized=True,verified=True))
        check('all_7185_exact_call_partitions',ledger==expected and len(ledger)==7185)
        runtime=read(fit/'runtime.json');check('runtime_flags',runtime['deterministic_algorithms'] is True and runtime['matmul_TF32'] is False and runtime['cudnn_TF32'] is False and runtime['AMP'] is False and runtime['CUBLAS_WORKSPACE_CONFIG']==':4096:8')
        stage='process_and_release'
        start=read(bind(experiment/'fit_process/start.json'));child=read(bind(experiment/'fit_process/child.json'));exit_record=read(bind(experiment/'fit_process/exit.json'))
        check('process_pid_links',start['wrapper_pid']==child['wrapper_pid']==exit_record['wrapper_pid'] and child['child_pid']==exit_record['child_pid'] and child['captured_handle_nonzero'] is True)
        check('start_request_identity',start['request_sha256']==sha(experiment/'training_request.json') and start['frozen_receipt_sha256']==sha(experiment/'training_frozen_inputs.json') and start['clearance_sha256']==sha(experiment/'training_clearance.json'))
        check('exit_known',exit_record['exit_known'] is True and exit_record['child_started'] is True and exit_record['all_postrun_pins_exact'] is True)
        for filename in ('prerun_pins.json','postrun_pins.json'):
            saved_pins=read(bind(experiment/'fit_process'/filename));check(filename+'_all_exact',saved_pins['all_exact'] is True)
            required_pins=dict(frozen['input_sha256']);required_pins.update({str(Path(frozen['source_directory'])/n):d for n,d in frozen['source_sha256'].items()})
            required_pins.update({str(experiment/'training_request.json'):sha(experiment/'training_request.json'),str(experiment/'training_frozen_inputs.json'):sha(experiment/'training_frozen_inputs.json'),str(experiment/'training_clearance.json'):sha(experiment/'training_clearance.json'),clearance['review_path']:clearance['review_sha256']})
            canonical={normalized_path(p):d for p,d in required_pins.items()};actual={normalized_path(p):v for p,v in saved_pins['files'].items()}
            check(filename+'_exact_membership',set(actual)==set(canonical))
            for p,d in canonical.items():check(filename+':'+p,actual[p]['expected']==actual[p]['actual']==d and actual[p]['matched'] is True)
        if qualified:
            check('successful_exit',exit_record['raw_python_exit_code']==exit_record['exit_code']==0 and exit_record['error'] is None)
            check('no_failure_capsule',(fit/'failure.json').exists() is False)
        else:
            failure=read(bind(fit/'failure.json'))
            check('preserved_numerical_failure',failure['error']=="ValueError('Final same-weight FP64 CPU/GPU/ORT preclamp parity failed; no release.')" and failure['optimization_completed'] is True and failure['final_export_diagnostics_completed'] is True)
            check('numerical_failure_exit',exit_record['raw_python_exit_code']!=0 and exit_record['exit_code']!=0)
        literal_roles=dict(fit_report=fit/'report.json',checkpoint=fit/'student_head.pt',head=fit/'student_head.onnx',normalization=fit/'normalization.npz',
            training_manifest=experiment/'training_frozen_inputs.json',training_request=experiment/'training_request.json',export_manifest=fit/'output_manifest.json',coefficient=fit/'coefficient.json',
            source_checkpoint=Path(source_subject['path']),full_state_generation_request=Path(request['full_state_paths']['request']),full_state_generation_report=Path(request['full_state_paths']['report']),
            full_state_data_audit=Path(request['subjects']['full_state_root_audit']['path']),full_state_data_owner=Path(request['subjects']['full_state_root_owner']['path']))
        for role,path in literal_roles.items():
            subject=audit_request['subjects'][role];check('audit_request_literal_role:'+role,Path(subject['path']).resolve()==path.resolve() and subject['sha256']==sha(path));bind(path,subject['sha256'])
        stage='final_rehash'
        for path,digest in list(tracked.items()):check('final_rehash:'+path,sha(path)==digest)
        result=dict(passed=qualified,evidence_audit_passed=True,export_qualified=qualified,canonical_evaluation_cleared=False,checks=len(checked),
            training_request_sha256=sha(experiment/'training_request.json'),fit_report_sha256=sha(fit/'report.json'),maximum_preclamp_rad=maximum,
            ordinary_final_step=65000,optimizer_step=10000,calibration_coefficient=weight,metrics=calculated,graph=graph_report,
            source_request_sha256=request_sha,audit_request_sha256=request_sha,direct_subject_sha256={role:sha(path) for role,path in literal_roles.items()},input_sha256=tracked,
            model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,
            limitations=['Saved gradients and predictions are checked for algebra and identity, without recomputing their model authenticity.',
                'Nominal float32 reductions use independently widened reduction with declared3e-7 relative roundoff tolerance; other metric algebra5e-12.',
                'This audit does not prove unsaved optimizer update gradients or closed-loop stability. Process absence is a separate owner qualification.'])
    except BaseException as error:
        result=dict(passed=False,evidence_audit_passed=False,export_qualified=False,canonical_evaluation_cleared=False,stage=stage,error=repr(error),
            checks=len(checked),input_sha256=tracked,model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0)
        raise
    finally:
        write(output/'checks.json',checked);write(output/'report.json',result)
    print(json.dumps(dict(evidence_audit_passed=True,export_qualified=result['export_qualified'],checks=len(checked),report_sha256=sha(output/'report.json'))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True);p.add_argument('--request-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.request,a.request_sha256,a.output)
