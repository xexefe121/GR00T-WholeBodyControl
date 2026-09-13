"""Independent final-fit evidence audit; no model/ORT/native/optimizer execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from audit_math import balanced_moments, phase_indices, summarize, targets
from audit_restoration import differences


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def run(experiment, output):
    fit=experiment/'fit';output.mkdir(exist_ok=False)
    sources=[Path(__file__),Path(__file__).with_name('audit_math.py'),Path(__file__).with_name('audit_restoration.py')]
    tracked={str(p):sha(p) for p in sources}
    checks=[]
    expected_failed_release_gates={'ordinary_final_completed','report_export_pass','all_export_parity_within_selected_tolerance'}

    def bind(path):
        path=Path(path);key=str(path);actual=sha(path)
        if key in tracked:assert tracked[key]==actual,key
        else:tracked[key]=actual
        return path

    def check(name,condition,detail=None):
        checks.append(dict(name=name,passed=bool(condition),detail=detail))
        if not condition and name not in expected_failed_release_gates:raise AssertionError(name)

    def exact(name,a,b):
        a,b=np.asarray(a),np.asarray(b)
        check(name,a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes())

    def close(name,a,b,rtol=5e-12,atol=1e-14):
        a,b=np.asarray(a),np.asarray(b)
        check(name,a.shape==b.shape and bool(np.allclose(a,b,rtol=rtol,atol=atol,equal_nan=False)),
              {'rtol':rtol,'atol':atol,'maximum_absolute_difference':float(np.max(np.abs(a-b))) if a.shape==b.shape and a.size else None})

    def compare(name,actual,expected):
        if isinstance(expected,dict):
            check(name+'.keys',set(actual)==set(expected))
            for key,value in expected.items():compare(name+'.'+key,actual[key],value)
        elif isinstance(expected,list):
            check(name+'.length',len(actual)==len(expected))
            for i,value in enumerate(expected):compare(name+f'[{i}]',actual[i],value)
        elif isinstance(expected,float):
            tolerant=name.endswith('normalized_MSE') or name.endswith('nominal_objective')
            close(name,actual,expected,rtol=3e-7 if tolerant else 5e-12)
        else:check(name,actual==expected)

    def npy(path):return np.load(bind(path),allow_pickle=False)

    def archive(path):
        with np.load(bind(path),allow_pickle=False) as values:return {k:values[k].copy() for k in values.files}

    try:
        request=read(bind(experiment/'training_request.json'));frozen=read(bind(experiment/'training_frozen_inputs.json'))
        report=read(bind(fit/'report.json'))
        check('completed_optimization_and_diagnostics_preserved',report['optimization_completed'] is True and report['final_export_diagnostics_completed'] is True and report['ordinary_final_step']==55000 and report['additional_updates']==50000)
        check('failed_release_status_preserved',report['completed'] is False and report['numerical_gate_passed'] is False and report['export_parity_passed'] is False)
        check('ordinary_final_completed',report['completed'] and report['optimization_completed'] and report['ordinary_final_step']==55000)
        check('report_export_pass',report['export_parity_passed'])
        check('request_identity',sha(experiment/'training_request.json')==frozen['training_request_sha256'])
        clearance=read(bind(experiment/'training_clearance.json'))
        check('clearance_subjects',clearance['approved'] is True and clearance['request_sha256']==sha(experiment/'training_request.json') and clearance['frozen_receipt_sha256']==sha(experiment/'training_frozen_inputs.json'))
        launch_review=read(bind(clearance['review_path']))
        check('clearance_review_identity',sha(clearance['review_path'])==clearance['review_sha256'] and launch_review[clearance['review_pass_field']] is True)
        check('clearance_launcher_identity',sha(bind(clearance['launcher_path']))==clearance['launcher_sha256'])
        for name,path in [('training_request',experiment/'training_request.json'),('frozen_inputs',experiment/'training_frozen_inputs.json'),('launcher',Path(clearance['launcher_path']))]:
            subject=launch_review['subjects'][name]
            check('direct_launch_review_subject.'+name,Path(subject['path']).resolve()==path.resolve() and subject['sha256']==sha(path))
        for path,digest in frozen['input_sha256'].items():check('frozen_input:'+path,sha(bind(path))==digest)
        for name,digest in frozen['source_sha256'].items():check('frozen_source:'+name,sha(bind(Path(frozen['source_directory'])/name))==digest)
        for path in fit.iterdir():
            if path.is_file() and path.name not in ('progress.json','progress.json.tmp'):bind(path)
        carried_request=read(bind(fit/'request.json'))
        expected_carried=dict(**request,source_receipt_sha256=sha(experiment/'training_frozen_inputs.json'),
            clearance_sha256=sha(experiment/'training_clearance.json'),normalization_sha256=sha(fit/'normalization.npz'),initialization_sha256=sha(fit/'initialization.pt'))
        compare('carried_fit_request',carried_request,expected_carried)
        paths=request['paths'];centers=archive(paths['centers']);pico=archive(paths['pico']);walk=archive(paths['walk002'])
        selected=np.concatenate([np.arange(52),np.arange(75,1023)]).astype(np.int64)
        features=np.concatenate([a['features'][:,selected] for a in (centers,pico,walk)])
        teacher=np.concatenate([a['expert_target'] for a in (centers,pico,walk)])
        exact('retained_feature_dtype_shape',np.asarray(features.shape,np.int64),np.array([9904,1000],np.int64))
        check('teacher_is_absolute_float64',teacher.dtype==np.float64 and teacher.shape==(9904,23))
        contract=read(bind(paths['contract']));default=np.asarray(contract['default_q'],np.float64)
        limits=np.asarray(contract['joint_limits'],np.float64);span=centers['joint_span']
        exact('native_rounded_span',span,(limits[:,1]-limits[:,0]).astype(np.float32))
        manifest=read(bind(paths['physical_manifest']));physical_dir=Path(paths['physical_manifest']).parent
        def physical_array(key):
            spec=manifest['arrays'][key];path=bind(physical_dir/spec['path'])
            check('physical_array_bound:'+key,sha(path)==spec['sha256'])
            return np.load(path,allow_pickle=False)
        physical_teacher=physical_array('label_fixed_map_target')
        physical_dataset=np.repeat(np.arange(3,dtype=np.int64),1018)
        physical_control=np.tile(np.arange(251,1269,dtype=np.int64),3)
        successor=physical_dataset*1019+physical_control-250
        exact('physical_dataset',physical_array('dataset'),physical_dataset)
        exact('physical_successor_control',physical_array('successor_control'),physical_control)
        exact('physical_successor_index',physical_array('successor_center_index'),successor)
        check('all_physical_labels_qualified',bool(physical_array('label_valid').all()))
        flags={name:physical_array(name) for name in ('policy_native_target_clipped','teacher_feedback_clipped','teacher_native_clipped','successor_replan_boundary','successor_zero_gain')}
        velocity_teacher=npy(paths['velocity_target']).reshape(140622,23)
        data=dict(default=default,span=span,limits=limits,nominal_teacher=teacher,velocity_teacher=velocity_teacher,
                  physical_teacher=physical_teacher,physical_successor=successor,physical_flags=flags)
        norm=archive(fit/'normalization.npz')
        exact('saved_feature_indices',norm['retained_feature_indices'],selected)
        for key,value in dict(joint_span=span,runtime_joint_span=span.astype(np.float64),default_q=default,joint_limits=limits).items():exact('normalization.'+key,norm[key],value)
        mean64,var64,_,_=balanced_moments(features,phase_indices())
        close('independent_weighted_mean',norm['weighted_mean64'],mean64,rtol=1e-11,atol=1e-12)
        close('independent_weighted_variance',norm['weighted_variance64'],var64,rtol=1e-11,atol=1e-12)
        exact('mean_cast',norm['feature_mean'],norm['weighted_mean64'].astype(np.float32))
        exact('std_floor_cast',norm['feature_std'],np.maximum(np.sqrt(norm['weighted_variance64']),.05).astype(np.float32))
        exact('normalized_absolute_labels',npy(fit/'nominal_normalized_labels.npy'),((teacher-default)/span.astype(np.float64)).astype(np.float32))
        ids=archive(fit/'data_identities.npz')
        dataset=np.concatenate([np.repeat(np.arange(3,dtype=np.int64),1019),np.full(5980,3,np.int64),np.full(867,4,np.int64)])
        control=np.concatenate([np.tile(np.arange(250,1269,dtype=np.int64),3),np.arange(250,6230,dtype=np.int64),np.arange(250,1117,dtype=np.int64)])
        phase=np.concatenate([np.full(len(group),i%3,np.int64) for i,group in enumerate(phase_indices())])
        exact('source_centers_dataset',centers['dataset'],dataset[:3057])
        exact('source_centers_controls',centers['control'],control[:3057])
        exact('source_centers_frames',centers['source_frame'],control[:3057]+11)
        for name,values,start,end in [('pico',pico,3057,9037),('walk002',walk,9037,9904)]:
            exact('source_'+name+'_controls',values['control'],control[start:end])
            exact('source_'+name+'_frames',values['source_frame'],control[start:end]+11)
            exact('source_'+name+'_phase',values['phase'],phase[start:end])
        for key,value in dict(dataset=dataset,control=control,phase=phase,source_frame=control+11,center_to_nominal=np.arange(3057,dtype=np.int64),physical_successor=successor,physical_dataset=physical_dataset,physical_control=physical_control).items():exact('identity.'+key,ids[key],value)
        exact('all_28800000_repeated_sampled_centers',npy(fit/'replayed_center_rows.npy'),np.tile(npy(paths['sampled_rows']),(10,1)))
        exact('all_28800000_repeated_sampled_axes',npy(fit/'replayed_axes.npy'),np.tile(npy(paths['sampled_axes']),(10,1)))
        progress=npy(fit/'training_progress.npy');check('50000_complete_finite_loss_records',progress.shape==(50000,6) and np.isfinite(progress).all())
        check('nonnegative_objectives_and_gradient_norm',bool((progress[:,:4]>=0).all() and (progress[:,5]>=0).all()))
        for index in range(50000):
            rate=3e-6+.5*(3e-5-3e-6)*(1+math.cos(math.pi*index/49999))
            check(f'learning_rate_{index+1}',progress[index,4]==rate)
        nc=npy(fit/'nominal_cell_losses.npy');vc=npy(fit/'velocity_cell_losses.npy');pc=npy(fit/'physical_cell_losses.npy')
        check('saved_cell_shapes',nc.shape==(50000,15) and vc.shape==pc.shape==(50000,9))
        close('nominal_loss_composition',progress[:,0],nc.mean(axis=1),rtol=3e-7)
        close('velocity_loss_composition',progress[:,1],vc.mean(axis=1))
        close('physical_loss_composition',progress[:,2],pc.mean(axis=1))
        close('three_equal_coefficients',progress[:,3],progress[:,:3].sum(axis=1))
        outputs={label:{corpus:npy(fit/f'{label}_{corpus}.npy') for corpus in ('nominal','velocity','physical')} for label in ('initial_GPU','final_GPU','final_CPU','final_ORT')}
        for label,values in outputs.items():
            for corpus,count in [('nominal',9904),('velocity',140622),('physical',3054)]:
                check(label+'.'+corpus+'.schema',values[corpus].shape==(count,23) and values[corpus].dtype==np.float32 and np.isfinite(values[corpus]).all())
        calculated={}
        for label,filename in [('initial_GPU','initial_metrics.json'),('final_GPU','final_metrics.json')]:
            values=outputs[label];result=summarize(values['nominal'],values['velocity'],values['physical'],data);calculated[label]=result
            saved=read(bind(fit/filename));compare(label+'.metrics',saved,result)
            compare(label+'.report_metrics',report['initial_metrics' if label=='initial_GPU' else 'final_metrics'],saved)
        numerical=read(bind(fit/'export_parity.json'));maximum=0.
        for corpus in ('nominal','velocity','physical'):
            raw={label:targets(outputs[label][corpus],default,span,limits)[0] for label in ('final_GPU','final_CPU','final_ORT')}
            for left,right,name in [('final_GPU','final_CPU','GPU_CPU'),('final_GPU','final_ORT','GPU_ORT'),('final_CPU','final_ORT','CPU_ORT')]:
                value=float(np.max(np.abs(raw[left]-raw[right])));maximum=max(maximum,value)
                check('parity.'+corpus+'_'+name,value==numerical['comparisons'][corpus+'_'+name])
        check('saved_parity_maximum_and_tolerance_identity',maximum==numerical['maximum_preclamp_rad'] and numerical['tolerance_rad']==1e-5)
        check('saved_parity_failure_classification',numerical['passed'] is False and math.isfinite(maximum) and maximum>1e-5 and numerical['nonfinite_comparisons']==[])
        check('all_export_parity_within_selected_tolerance',maximum==numerical['maximum_preclamp_rad'] and maximum<=1e-5 and numerical['tolerance_rad']==1e-5)
        compare('report_parity',report['export_parity'],numerical)
        for key,value in {'training_head_rows_attempted':705500000,'training_head_rows_returned':705500000,'diagnostic_torch_rows_attempted':460740,'diagnostic_torch_rows_returned':460740,'ORT_calls_attempted':601,'ORT_calls_returned':601}.items():check('counter.'+key,report['counters'][key]==value)
        import torch
        import onnx
        from onnx import numpy_helper
        checkpoint=torch.load(bind(fit/'student_head.pt'),map_location='cpu',weights_only=True)
        check('checkpoint_direct_output_contract',checkpoint['kind']=='direct_absolute_native23_target' and checkpoint['output']=='normalized_absolute_target' and checkpoint['seed']==20260911 and checkpoint['ordinary_final_step']==55000 and checkpoint['additional_updates']==50000)
        compare('checkpoint_request',checkpoint['request'],request)
        old_fit=experiment.parent/'direct_target_student_v1/fit'
        old_checkpoint_path=bind(old_fit/'student_head.pt')
        check('exact_selected_source_checkpoint',sha(old_checkpoint_path)=='ad6d657affba0796e7313d85ace240cd31d46e188c577da049880a7a031aecba')
        old_checkpoint=torch.load(old_checkpoint_path,map_location='cpu',weights_only=True)
        restored=torch.load(bind(fit/'initialization.pt'),map_location='cpu',weights_only=True)
        check('restoration_source_subject',restored['source_checkpoint_sha256']==sha(old_checkpoint_path))
        for name in ('actor_state','optimizer_state'):
            problems=differences(restored[name],old_checkpoint[name],name)
            check('restored_start_exact.'+name,not problems,problems)
        problems=differences(restored['rng_after_restoration'],old_checkpoint['rng'],'rng')
        check('restored_all_rng_exact',not problems,problems)
        problems=differences(checkpoint['rng'],old_checkpoint['rng'],'final_rng')
        check('no_new_rng_draws_in_fixed_continuation',not problems,problems)
        old_norm=archive(old_fit/'normalization.npz')
        check('normalization_keys_unchanged',set(old_norm)==set(norm))
        for key in norm:exact('source_normalization_exact.'+key,norm[key],old_norm[key])
        for corpus in ('nominal','velocity','physical'):
            exact('initial_GPU_matches_old_final_GPU.'+corpus,outputs['initial_GPU'][corpus],npy(old_fit/f'final_GPU_{corpus}.npy'))
        compare('initial_metrics_match_old_final',read(bind(fit/'initial_metrics.json')),read(bind(old_fit/'final_metrics.json')))
        check('selected_additional_budget',request['updates']==50000)

        optimization=read(bind(fit/'optimization_completed.json'))
        check('optimization_receipt_subject',optimization['optimization_completed'] is True and optimization['ordinary_final_step']==55000 and optimization['checkpoint_sha256']==sha(fit/'student_head.pt') and optimization['export_validation_pending'] is True)
        for key in ('feature_mean','feature_std','joint_span','runtime_joint_span','default_q','joint_limits','retained_feature_indices'):exact('checkpoint.'+key,checkpoint[key].numpy(),norm[key])
        optimizer=checkpoint['optimizer_state'];check('six_optimizer_states',len(optimizer['state'])==6)
        for key,state in optimizer['state'].items():
            check('optimizer_step:'+str(key),float(state['step'])==55000)
            check('optimizer_moments_finite:'+str(key),all(bool(torch.isfinite(state[name]).all()) for name in ('exp_avg','exp_avg_sq')))
        group=optimizer['param_groups'][0]
        check('optimizer_contract',len(optimizer['param_groups'])==1 and group['lr']==3e-6 and group['weight_decay']==1e-5 and group['foreach'] is False and group['fused'] is False)
        graph=onnx.load(bind(fit/'student_head.onnx'));onnx.checker.check_model(graph)
        check('onnx_opset_ir',graph.ir_version==8 and [(o.domain,o.version) for o in graph.opset_import]==[('',17)])
        check('onnx_operations',[node.op_type for node in graph.graph.node]==['Sub','Div','MatMul','Add','Elu','MatMul','Add','Elu','MatMul','Add'])
        check('onnx_io_names',[v.name for v in graph.graph.input]==['features'] and [v.name for v in graph.graph.output]==['normalized_target'])
        for tensor,width,name in [(graph.graph.input[0],1000,'input'),(graph.graph.output[0],23,'output')]:
            kind=tensor.type.tensor_type;dims=list(kind.shape.dim)
            check('onnx_'+name+'_float32_batch_shape',kind.elem_type==onnx.TensorProto.FLOAT and len(dims)==2 and not dims[0].HasField('dim_value') and dims[0].dim_param=='' and dims[1].HasField('dim_value') and dims[1].dim_value==width)
        wires=[('Sub',['features','mean'],['center']),('Div',['center','std'],['normalized']),
            ('MatMul',['normalized','w0'],['m0']),('Add',['m0','b0'],['a0']),('Elu',['a0'],['e0']),
            ('MatMul',['e0','w1'],['m1']),('Add',['m1','b1'],['a1']),('Elu',['a1'],['e1']),
            ('MatMul',['e1','w2'],['m2']),('Add',['m2','b2'],['normalized_target'])]
        for index,(node,(op,inputs,outputs_expected)) in enumerate(zip(graph.graph.node,wires)):
            check(f'onnx_node_{index}_wiring',node.op_type==op and list(node.input)==inputs and list(node.output)==outputs_expected and node.domain=='')
            attributes=list(node.attribute)
            check(f'onnx_node_{index}_attributes',len(attributes)==1 and attributes[0].name=='alpha' and attributes[0].type==onnx.AttributeProto.FLOAT and attributes[0].f==1.0 if op=='Elu' else len(attributes)==0)
        tensors={item.name:numpy_helper.to_array(item) for item in graph.graph.initializer}
        check('onnx_initializer_keys',set(tensors)=={'mean','std','w0','b0','w1','b1','w2','b2'})
        exact('onnx_mean',tensors['mean'],norm['feature_mean']);exact('onnx_std',tensors['std'],norm['feature_std'])
        for layer,name in enumerate((0,2,4)):
            check(f'fixed_layer_{layer}_dimensions',checkpoint['actor_state'][f'{name}.weight'].shape==((256,1000),(256,256),(23,256))[layer] and checkpoint['actor_state'][f'{name}.bias'].shape==((256,),(256,),(23,))[layer])
            exact(f'onnx_weight_{layer}',tensors[f'w{layer}'],checkpoint['actor_state'][f'{name}.weight'].numpy().T.copy())
            exact(f'onnx_bias_{layer}',tensors[f'b{layer}'],checkpoint['actor_state'][f'{name}.bias'].numpy())
        check('report_checkpoint_hash',sha(fit/'student_head.pt')==report['checkpoint_sha256'])
        check('report_export_hash',sha(fit/'student_head.onnx')==report['onnx_sha256'])
        for path,digest in tracked.items():check('final_rehash:'+path,sha(path)==digest)
        failed_release_gates=[c['name'] for c in checks if not c['passed']]
        check('exact_expected_failed_release_gates',set(failed_release_gates)==expected_failed_release_gates and len(failed_release_gates)==3)
        result=dict(passed=False,evidence_audit_passed=True,optimization_evidence_verified=True,export_qualified=False,canonical_cleared=False,failed_release_gates=failed_release_gates,checks=len(checks),initial_metrics=calculated['initial_GPU'],final_metrics=calculated['final_GPU'],maximum_preclamp_export_error_rad=maximum,
                    input_sha256=tracked,task_model_calls=0,ORT_calls=0,optimizer_updates=0,native_steps=0,
                    limitations=['saved losses cannot reconstruct unsaved gradients or prove connected stability','nominal float32 mean independently checked with3e-7 relative roundoff tolerance','restored model/optimizer/RNG and initial predictions are checked exactly; CUDA execution flags covered by trainer source/runtime review'])
    except BaseException as error:
        result=dict(passed=False,error=repr(error),checks=len(checks),input_sha256=tracked,task_model_calls=0,ORT_calls=0,optimizer_updates=0,native_steps=0)
        raise
    finally:
        (output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'evidence_audit_passed':True,'export_qualified':False,'failed_release_gates':failed_release_gates,'checks':len(checks),'report_sha256':sha(output/'report.json')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.experiment,args.output)
