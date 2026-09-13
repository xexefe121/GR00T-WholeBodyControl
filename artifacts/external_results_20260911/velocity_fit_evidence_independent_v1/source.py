"""Saved-evidence audit of the fixed 65000-to-70000 fit; no model evaluation.

Reconstructs all sampling draws using a private Torch generator, checks original
normalization and optimizer counters, and recalculates core nominal/chord/export
metrics over all preserved predictions. This does not replay optimizer updates.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import onnx
from onnx import numpy_helper


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def archive(path):
    with np.load(path,allow_pickle=False) as data:
        return {key:data[key].copy() for key in data.files}


def exact(a,b,name):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),name


def local(value):
    value=str(value).replace('\\','/')
    if value.startswith('/mnt/'):
        value=value[5].upper()+':'+value[6:]
    return Path(value)


def main(args):
    assert sys.version_info[:3]==(3,10,11)
    assert np.__version__=='1.23.5' and torch.__version__=='2.10.0+cpu' and onnx.__version__=='1.22.0'
    torch.set_num_threads(1)
    experiment=args.experiment
    fit=experiment/'fit'
    prior=experiment.parent/'fast_controller_phase_fit_v1/fit'
    paths=[fit/name for name in ('student_head.pt','student_head.onnx','training_arrays.npz',
        'sampled_center_rows.npy','sampled_axes.npy','initial_predictions.npz','final_predictions.npz',
        'initial_chord_metrics.json','final_chord_metrics.json','initial_nominal_metrics.json',
        'final_nominal_metrics.json','report.json')]
    paths += [prior/'student_head.pt',prior/'student_head.onnx',prior/'teacher_fit.npz',prior/'report.json',
        experiment/'generation/centers.npz',experiment/'generation/base_target.npy',
        experiment/'generation/teacher_target.npy',experiment/'training_frozen_inputs.json']
    paths += [fit/'request.json',fit/'restoration65000_parity.json',fit/'optimization_completed.json',
              experiment/'training_clearance.json']
    pins={str(path):sha(path) for path in paths}
    assert sha(experiment/'training_frozen_inputs.json')=='1c9854317d178959c40d59aaa915746a8bd101f0d44e46237c9d1fd2a7a46ba2'
    receipt=read(experiment/'training_frozen_inputs.json')
    source_directory=local(receipt['source_directory'])
    for name,digest in receipt['source_sha256'].items():
        path=source_directory/name
        assert sha(path)==digest,name
        pins[str(path)]=digest
    for name,digest in receipt['input_sha256'].items():
        path=local(name)
        assert sha(path)==digest,name
        pins[str(path)]=digest
    clearance=read(experiment/'training_clearance.json')
    assert clearance['approved'] is True
    assert clearance['frozen_receipt_sha256']==sha(experiment/'training_frozen_inputs.json')
    for kind in ('source','dataset'):
        path=local(clearance[kind+'_review_path'])
        assert sha(path)==clearance[kind+'_review_sha256']
        pins[str(path)]=sha(path)
    request=dict(kind='independent_saved_velocity_fit_evidence',source_sha256=sha(__file__),
                 input_sha256=pins,model_evaluations=0,optimizer_updates=0,physics_steps=0,
                 sampling_draws=45000,sampling_row_axis_pairs=2880000,
                 runtime=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,onnx=onnx.__version__,torch_threads=1))
    (args.output/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    report=read(fit/'report.json')
    assert all(report[key] is True for key in ('completed','optimization_completed',
               'final_export_diagnostics_completed','numerical_gate_passed'))
    assert report['ordinary_final_step']==70000 and report['additional_updates']==5000
    assert report['head_ONNX_calls']==1126 and report['export_parity_passed'] is True
    assert sha(fit/'student_head.pt')==report['checkpoint_sha256']
    assert sha(fit/'student_head.onnx')==report['onnx_sha256']
    fit_request=read(fit/'request.json')
    assert (fit_request['first_step'],fit_request['last_step'],fit_request['updates'])==(65001,70000,5000)
    assert fit_request['frozen_receipt_sha256']==sha(experiment/'training_frozen_inputs.json')
    assert fit_request['clearance_sha256']==sha(experiment/'training_clearance.json')
    assert fit_request['prior_sha256']==sha(prior/'student_head.pt')
    optimization=read(fit/'optimization_completed.json')
    assert optimization['ordinary_final_step']==70000
    assert optimization['checkpoint_sha256']==sha(fit/'student_head.pt')
    assert optimization['training_arrays_sha256']==sha(fit/'training_arrays.npz')
    before=torch.load(prior/'student_head.pt',map_location='cpu',weights_only=True)
    after=torch.load(fit/'student_head.pt',map_location='cpu',weights_only=True)
    assert after['request']==fit_request
    assert before['completed_steps']==65000 and after['completed_steps']==70000
    assert len(before['optimizer_state']['state'])==len(after['optimizer_state']['state'])==6
    for key in ('feature_mean','feature_std','joint_span'):
        assert torch.equal(before[key],after[key]),key
    for state in after['optimizer_state']['state'].values():
        assert int(state['step'])==70000
        assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
    for parameter in after['actor_state'].values():
        assert torch.isfinite(parameter).all()
    assert any(not torch.equal(before['actor_state'][key],value) for key,value in after['actor_state'].items())
    exported=onnx.load(fit/'student_head.onnx')
    initial_export=onnx.load(prior/'student_head.onnx')
    tensors={value.name:numpy_helper.to_array(value) for value in exported.graph.initializer}
    assert set(tensors)=={'mean','std','span','w0','w1','w2','b0','b1','b2'}
    for key,saved in (('mean','feature_mean'),('std','feature_std'),('span','joint_span')):
        exact(tensors[key],after[saved].numpy(),'export '+key)
    for layer in range(3):
        exact(tensors['w'+str(layer)],after['actor_state'][str(layer*2)+'.weight'].numpy().T,'export weight')
        exact(tensors['b'+str(layer)],after['actor_state'][str(layer*2)+'.bias'].numpy(),'export bias')
    del exported.graph.initializer[:]
    del initial_export.graph.initializer[:]
    assert exported.SerializeToString(deterministic=True)==initial_export.SerializeToString(deterministic=True)
    assert before['rng']['numpy']==after['rng']['numpy']
    centers=archive(experiment/'generation/centers.npz')
    cells=[np.arange(dataset*1019+start,dataset*1019+stop) for dataset in range(3)
           for start,stop in ((0,100),(100,919),(919,1019))]
    exact(centers['control'],np.tile(np.arange(250,1269,dtype=np.int64),3),'control order')
    exact(centers['dataset'],np.repeat(np.arange(3,dtype=np.int64),1019),'dataset order')
    rows=np.load(fit/'sampled_center_rows.npy',allow_pickle=False)
    axes=np.load(fit/'sampled_axes.npy',allow_pickle=False)
    assert rows.shape==axes.shape==(5000,576)
    assert rows.dtype==np.int32 and axes.dtype==np.int8
    generator=torch.Generator(device='cpu')
    generator.set_state(before['rng']['torch'])
    for update in range(5000):
        for cell,ids in enumerate(cells):
            draws=torch.randint(0,len(ids)*23,(64,),generator=generator).numpy()
            section=slice(cell*64,(cell+1)*64)
            exact(rows[update,section],ids[draws//23].astype(np.int32),'sampled rows')
            exact(axes[update,section],(draws%23).astype(np.int8),'sampled axes')
    assert torch.equal(generator.get_state(),after['rng']['torch'])
    arrays=archive(fit/'training_arrays.npz')
    exact(arrays['global_step'],np.arange(65001,70001),'all5000 global steps')
    rates=np.asarray([3e-7+.5*(3e-6-3e-7)*(1+math.cos(math.pi*k/4999)) for k in range(5000)])
    exact(arrays['learning_rate'],rates,'exact cosine learning rate')
    for key in ('nominal_objective','sampled_chord_objective','training_objective'):
        assert arrays[key].shape==(5000,) and np.isfinite(arrays[key]).all() and np.all(arrays[key]>=0)
    exact(arrays['nominal_objective']+arrays['sampled_chord_objective'],arrays['training_objective'],'loss composition')
    assert all(group['lr']==rates[-1] and group['weight_decay']==1e-5 for group in after['optimizer_state']['param_groups'])
    span=before['joint_span'].numpy()
    exact(span,centers['joint_span'],'native span')
    labels=torch.from_numpy((centers['residual_rad']/span).astype(np.float32))
    base=np.load(experiment/'generation/base_target.npy',mmap_mode='r',allow_pickle=False)
    teacher=np.load(experiment/'generation/teacher_target.npy',mmap_mode='r',allow_pickle=False)
    limits=centers['joint_limits']
    summaries={}
    for stage in ('initial','final'):
        predictions=archive(fit/(stage+'_predictions.npz'))
        delta=predictions['predicted_delta']
        ort_delta=predictions['onnx_predicted_delta']
        assert delta.shape==ort_delta.shape==(143679,23)
        assert delta.dtype==ort_delta.dtype==np.float32
        assert all(np.isfinite(value).all() for value in predictions.values())
        parity=float(max(np.max(np.abs(delta-ort_delta)),np.max(np.abs(
            predictions['actual_seven_delta']-predictions['actual_seven_onnx_delta']))))
        assert parity<1e-5
        normalized=torch.from_numpy(predictions['normalized_centers'])
        assert normalized.shape==(3057,23)
        exact((normalized*torch.from_numpy(span)).numpy(),delta[:3057],'nominal normalized/delta')
        square=(normalized-labels)**2
        nominal=float(torch.stack([square[ids].mean() for ids in cells]).mean())
        nominal_report=read(fit/(stage+'_nominal_metrics.json'))
        assert nominal==nominal_report['nine_cell_nominal_objective']
        assert nominal==report[stage+'_nominal_objective']
        center_proposal=centers['base_target']+delta[:3057]
        probe_proposal=base+delta[3057:].reshape(3057,23,2,23)
        difference=(probe_proposal-center_proposal[:,None,None])-(teacher-centers['expert_target'][:,None,None])
        normalized_square=(difference/span)**2
        objectives=[float(normalized_square[ids].mean()) for ids in cells]
        chord=float(np.mean(objectives))
        chord_report=read(fit/(stage+'_chord_metrics.json'))
        assert chord==chord_report['full_nine_cell_chord_objective']==report[stage+'_chord_objective']
        probe_applied=np.clip(probe_proposal,limits[:,0],limits[:,1])
        for cell,ids in enumerate(cells):
            saved=chord_report['cells'][cell]
            assert saved['normalized_proposal_chord_MSE']==objectives[cell]
            assert saved['student_clipped_probes']==int(np.any(probe_proposal[ids]!=probe_applied[ids],axis=-1).sum())
            assert saved['student_clipped_components']==int(np.sum(probe_proposal[ids]!=probe_applied[ids]))
        target_error=np.clip(center_proposal,limits[:,0],limits[:,1])-centers['expert_target']
        target_rmse=float(np.sqrt(np.mean(target_error**2)))
        query250=float(np.sqrt(np.mean(target_error[2038]**2)))
        assert target_rmse==nominal_report['all']['applied_target_rmse_rad']
        assert query250==nominal_report['query250_target_rmse_rad']
        if stage=='initial':
            exact(delta[:3057],archive(prior/'teacher_fit.npz')['predicted_delta'],'original65000 nominal predictions')
            assert nominal==read(prior/'report.json')['final_nine_cell_objective']
        else:
            assert parity==report['final_head_parity']
        summaries[stage]=dict(nominal_objective=nominal,chord_objective=chord,
            nominal_target_rmse_rad=target_rmse,query250_target_rmse_rad=query250,export_max_delta_rad=parity)
    for path,digest in pins.items():
        assert sha(path)==digest,path
    assert sha(__file__)==request['source_sha256']
    result=dict(kind='independent_saved_velocity_fit_evidence',passed=True,all_inputs_unchanged=True,
        all2880000_sampler_pairs_and_final_Torch_RNG_exact=True,NumPy_RNG_unchanged=True,
        all5000_learning_rates_and_loss_compositions_exact=True,ordinary_final_optimizer_steps=70000,
        original_normalization_and_spans_exact=True,ONNX_parameters_exact_and_graph_structure_unchanged=True,
        initial_final_core_metrics_over_all_saved_predictions_exact=True,
        summaries=summaries,model_evaluations=0,optimizer_updates=0,physics_steps=0,
        limitation='Saved-evidence and reviewed-source audit; optimizer updates not replayed. Per-joint, fixed-pair, first24 and sensitivity report fields are outside this audit.',
        request_sha256=sha(args.output/'request.json'),source_sha256=sha(__file__))
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    try:
        main(args)
    except BaseException as error:
        import traceback
        (args.output/'failure.json').write_text(json.dumps(dict(passed=False,
            exception=type(error).__name__,message=str(error),traceback=traceback.format_exc()),indent=2)+'\n')
        raise
