"""One fixed nominal supervised fit; no cap, epoch, seed or model sweep."""
import json
from pathlib import Path
import time
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
import onnxruntime as ort
import torch
from student_linear_runtime import BASE,KIND,FEATURES,FROZEN_RECEIPT,assert_frozen,archive,sha

CONFIG=dict(seed=773,steps=1000,batch_size=256,learning_rate=3e-4,weight_decay=1e-5,
    hidden=[256,256],activation='ELU',output='linear23 times nativejointspan, no tanh or radius',
    feature_std_floor=.05,gradient_clip_norm=10.,device='cpu',threads=1,
    phase_sampling='64each initial250/acquisition100/source819/return100',
    validation='all1269 labels train; full closed-loop nominal rollout is behavioral test, no heldout-generalization claim')


def make_actor():
    model=torch.nn.Sequential(torch.nn.Linear(FEATURES,256),torch.nn.ELU(),torch.nn.Linear(256,256),torch.nn.ELU(),torch.nn.Linear(256,23))
    torch.nn.init.zeros_(model[-1].weight);torch.nn.init.zeros_(model[-1].bias)
    return model


def export(model,mean,std,span,path):
    initial=[numpy_helper.from_array(mean,'mean'),numpy_helper.from_array(std,'std'),numpy_helper.from_array(span,'span')]
    nodes=[helper.make_node('Sub',['features','mean'],['center']),helper.make_node('Div',['center','std'],['normalized'])]
    value='normalized'
    for number,index in enumerate((0,2,4)):
        layer=model[index]
        initial += [numpy_helper.from_array(layer.weight.detach().numpy().T.copy(),f'w{number}'),numpy_helper.from_array(layer.bias.detach().numpy().copy(),f'b{number}')]
        nodes += [helper.make_node('MatMul',[value,f'w{number}'],[f'm{number}']),helper.make_node('Add',[f'm{number}',f'b{number}'],[f'a{number}'])]
        value=f'a{number}'
        if number<2:
            nodes.append(helper.make_node('Elu',[value],[f'e{number}'],alpha=1.));value=f'e{number}'
    nodes.append(helper.make_node('Mul',[value,'span'],['delta_rad']))
    graph=helper.make_graph(nodes,'native23_linear_span_student',[helper.make_tensor_value_info('features',TensorProto.FLOAT,[None,FEATURES])],[helper.make_tensor_value_info('delta_rad',TensorProto.FLOAT,[None,23])],initial)
    model_proto=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]);model_proto.ir_version=8
    onnx.checker.check_model(model_proto);onnx.save(model_proto,str(path))


def main():
    assert_frozen();torch.set_num_threads(1);torch.manual_seed(CONFIG['seed']);np.random.seed(CONFIG['seed'])
    dest=BASE/'fit';dest.mkdir(exist_ok=True)
    assert not (dest/'student_head.pt').exists() and not (dest/'report.json').exists()
    parity=json.loads((BASE/'zero_parity/report.json').read_text())
    assert parity['all_control_state_history_rawaction_target_bitexact'] and parity['native2msstrict_pass']
    data=archive(BASE/'labels/labels.npz');receipt=json.loads((BASE/'labels/report.json').read_text())
    assert sha(BASE/'labels/labels.npz')==receipt['labels_sha256'] and receipt['samples']==1269
    X=data['features'].astype(np.float32);span=data['joint_span'].astype(np.float32)
    Y=(data['residual_rad']/span).astype(np.float32)
    mean=X.mean(0).astype(np.float32);std=np.maximum(X.std(0),CONFIG['feature_std_floor']).astype(np.float32)
    features=torch.from_numpy((X-mean)/std);labels=torch.from_numpy(Y)
    actor=make_actor();optimizer=torch.optim.AdamW(actor.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    strata=[np.arange(0,250),np.arange(250,350),np.arange(350,1169),np.arange(1169,1269)]
    request=dict(kind=KIND,configuration=CONFIG,labels_sha256=sha(BASE/'labels/labels.npz'),
        frozen_sources_sha256=sha(FROZEN_RECEIPT),joint_span=span.tolist(),samples=1269,
        no_dagger_labels=True,no_hardware=True,training_metric_is_not_validation=True)
    (dest/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    initial_zero=dest/'zero_head_training_initialization.onnx'
    export(actor,mean,std,span,initial_zero)
    assert sha(initial_zero)==sha(dest/'zero_head.onnx'),'training initialization differs from prefit zero parity'
    started=time.perf_counter();records=[]
    for step in range(1,CONFIG['steps']+1):
        picked=np.concatenate([np.random.choice(ids,64,replace=True) for ids in strata])
        prediction=actor(features[picked]);loss=torch.mean((prediction-labels[picked])**2)
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),CONFIG['gradient_clip_norm']);optimizer.step()
        if step%100==0:
            record=dict(step=step,normalized_residual_training_loss=float(loss),seconds=time.perf_counter()-started)
            records.append(record);print(json.dumps(record),flush=True)
    actor.eval()
    with torch.inference_mode():predicted=(actor(features)*torch.from_numpy(span)).numpy()
    predicted_target=np.clip(data['base_target']+predicted,data['joint_limits'][:,0],data['joint_limits'][:,1])
    errors=predicted_target-data['expert_target']
    residual_errors=predicted-data['residual_rad']
    summaries={}
    for name,indices in zip(('initial_entry','acquisition','source','return'),strata):
        e=errors[indices];r=residual_errors[indices]
        summaries[name]=dict(controls=len(indices),applied_target_rmse_rad=float(np.sqrt(np.mean(e**2))),
            applied_target_rmse_by_joint_rad=np.sqrt(np.mean(e**2,axis=0)).tolist(),
            applied_target_abs_error_p95_by_joint_rad=np.percentile(np.abs(e),95,axis=0).tolist(),
            preclip_residual_rmse_rad=float(np.sqrt(np.mean(r**2))),applied_target_max_error_rad=float(np.max(np.abs(e))))
    export(actor,mean,std,span,dest/'student_head.onnx')
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    session=ort.InferenceSession(str(dest/'student_head.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    exported=session.run(None,{'features':X})[0]
    max_delta=float(np.max(np.abs(exported-predicted)))
    assert max_delta<1e-5,(max_delta,'ONNX export mismatch')
    zero=ort.InferenceSession(str(dest/'zero_head.onnx'),sess_options=options,providers=['CPUExecutionProvider']).run(None,{'features':X})[0]
    np.testing.assert_array_equal(zero,np.zeros_like(zero))
    np.testing.assert_array_equal(data['base_target']+zero,data['base_target'])
    np.savez_compressed(dest/'teacher_fit.npz',predicted_delta=predicted,predicted_applied_target=predicted_target,
        expert_target=data['expert_target'],applied_error=errors,feature_mean=mean,feature_std=std,source_frame=data['source_frame'])
    checkpoint=dict(kind=KIND,actor_state=actor.state_dict(),feature_mean=torch.from_numpy(mean),feature_std=torch.from_numpy(std),joint_span=torch.from_numpy(span),request=json.loads(json.dumps(request)))
    torch.save(checkpoint,dest/'student_head.pt')
    result=dict(kind=KIND,steps=CONFIG['steps'],elapsed_seconds=time.perf_counter()-started,teacher_state_training_metrics=summaries,
        ONNX_vs_Torch_max_delta_rad=max_delta,zero_head_all1269_teacher_states_exact=True,
        network_training_complete=True,closed_loop_assessment_pending=True,no_generalization_claim=True,
        checkpoints={p.name:sha(p) for p in (dest/'zero_head.onnx',dest/'student_head.onnx',dest/'student_head.pt')})
    (dest/'metrics.json').write_text(json.dumps(records,indent=2)+'\n');(dest/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
