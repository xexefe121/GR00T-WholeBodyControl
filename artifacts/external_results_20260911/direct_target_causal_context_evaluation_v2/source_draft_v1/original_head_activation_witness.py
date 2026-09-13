"""Exactly one separately selected WSL batch-one head call after final reviews."""
from pathlib import Path
import json
import sys
import traceback
import numpy as np
from evaluation_gate import bound_file,field,has_hash,local,read,sha
from proposal_evidence import exact

BASE=Path(__file__).resolve().parent.parent
CENTERS_SHA='5b07595d07e262f2ad236565ec62483d600843fe27a9f4a781974dac4b0608c7'
LABEL_SHA='bdcba927cd737a2ba74d75d7b31c427a5bfd4df7437983d13c391f9ca166f749'


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def atomic(path,value):
    temporary=Path(path).with_suffix('.tmp');write(temporary,value);temporary.replace(path)


def require_witness_ready(base):
    path=Path(base)/'witness_binding.json'
    if not path.is_file():raise ValueError('Final75000 head witness is not bound; no call is permitted.')
    binding=read(path)
    assert binding['root_authorized_single_head_witness'] is True
    assert binding['expected_head_calls']==1 and binding['BFM_inference_authorized'] is False
    assert binding['physics_authorized'] is False and binding['fitting_authorized'] is False
    paths={name:bound_file(binding[name]) for name in ('head','checkpoint','fit_report','training_manifest','centers','query250_labels')}
    assert binding['centers']['sha256']==CENTERS_SHA and binding['query250_labels']['sha256']==LABEL_SHA
    for entry in binding['input_files']:bound_file(entry)
    fit=read(paths['fit_report'])
    assert fit['ordinary_final_step']==75000 and fit['additional_updates']==5000
    for key in ('optimization_completed','final_export_diagnostics_completed','numerical_gate_passed','export_parity_passed'):
        assert fit[key] is True,key
    assert fit['checkpoint_sha256']==binding['checkpoint']['sha256'] and fit['onnx_sha256']==binding['head']['sha256']
    subjects=dict(dataset=CENTERS_SHA,fit=binding['fit_report']['sha256'],export=binding['head']['sha256'],source=sha(__file__))
    for role,digest in subjects.items():
        entry=binding['reviews'][role];review=read(bound_file(entry))
        assert field(review,entry['pass_field']) is True and has_hash(review,digest),role
    with np.load(paths['centers'],allow_pickle=False) as data:
        assert data['dataset'][2038]==2 and data['control'][2038]==250 and data['source_frame'][2038]==261
        center={key:data[key][2038].copy() for key in ('features','base_target','previous_action','history','state','qpos','qvel')}
    with np.load(paths['query250_labels'],allow_pickle=False) as data:
        assert data['control'][0]==250 and data['source_frame'][0]==261
        for key in ('features','base_target','previous_action','history','state'):
            assert exact(center[key],data[key][0]),key
        assert exact(center['qpos'],data['teacher_qpos'][0]) and exact(center['qvel'],data['teacher_qvel'][0])
    assert center['features'].shape==(1069,) and center['features'].dtype==np.float32
    return binding,paths,center,sha(path)


def main():
    binding,paths,center,binding_sha=require_witness_ready(BASE)
    output=BASE/'head_witness';output.mkdir(exist_ok=False)
    attempted=0;context={key:value.copy() for key,value in center.items()}
    atomic(output/'attempt.json',dict(expected_head_calls=1,attempted_head_calls=0,binding_sha256=binding_sha))
    try:
        assert sys.platform!='win32','Witness must use the evaluator WSL runtime.'
        dependencies=local(binding['onnx_dependencies'])
        sys.path.insert(0,str(dependencies))
        import onnxruntime as ort
        assert ort.__version__=='1.23.2' and np.__version__=='1.26.4'
        binary=list((Path(ort.__file__).parent/'capi').glob('*pybind11_state*.so'))
        assert len(binary)==1
        runtime_sha=sha(binary[0])
        assert any(item['sha256']==runtime_sha for item in binding['input_files'])
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        head=ort.InferenceSession(str(paths['head']),sess_options=options,providers=['CPUExecutionProvider'])
        inputs=head.get_inputs();assert len(inputs)==1 and inputs[0].name=='features'
        features=center['features'].copy()
        attempted=1
        atomic(output/'attempt.json',dict(expected_head_calls=1,attempted_head_calls=attempted,binding_sha256=binding_sha))
        returned=head.run(None,{'features':features[None]})
        for index,value in enumerate(returned):context['returned_'+str(index)]=np.asarray(value).copy()
        assert len(returned)==1 and returned[0].shape==(1,23) and returned[0].dtype==np.float32
        delta=returned[0][0].copy();assert np.isfinite(delta).all()
        _,_,_,current_sha=require_witness_ready(BASE);assert current_sha==binding_sha
        np.savez_compressed(output/'witness.npz',features=features,onnx_delta=delta,
            control=np.asarray(250,np.int64),source_frame=np.asarray(261,np.int64),
            head_sha256=np.asarray(binding['head']['sha256']),centers_sha256=np.asarray(CENTERS_SHA),
            query250_labels_sha256=np.asarray(LABEL_SHA),runtime_binary_sha256=np.asarray(runtime_sha))
        report=dict(kind='one_separately_selected_WSL_final75000_head_activation_witness',pass_all=True,
            head_sha256=binding['head']['sha256'],witness_sha256=sha(output/'witness.npz'),
            ordinary_final_step=75000,control=250,source_frame=261,centers_row=2038,
            exact_original_query250_inputs=True,expected_head_calls=1,attempted_head_calls=attempted,
            onnxruntime_version=ort.__version__,intra_op_threads=1,inter_op_threads=1,
            execution_mode='ORT_SEQUENTIAL',execution_provider='CPUExecutionProvider',pinned_WSL_runtime=True,
            runtime_binary_sha256=runtime_sha,runtime_binary_path=str(binary[0]),
            features_shape=list(features.shape),features_dtype=str(features.dtype),
            output_shape=list(delta.shape),output_dtype=str(delta.dtype),
            binding_sha256=binding_sha,source_sha256=sha(__file__),fit_head_calls_repeated=0,
            BFM_inference_calls=0,physics_steps=0,fitting_launched=False)
        write(output/'report.json',report)
        print(json.dumps(dict(pass_all=True,attempted_head_calls=attempted,witness_sha256=report['witness_sha256'])),flush=True)
    except BaseException as error:
        np.savez_compressed(output/'failed_inputs_outputs.npz',**context)
        write(output/'failure.json',dict(pass_all=False,error=repr(error),traceback=traceback.format_exc(),
            expected_head_calls=1,attempted_head_calls=attempted,binding_sha256=binding_sha,
            BFM_inference_calls=0,physics_steps=0,fitting_launched=False))
        raise


if __name__=='__main__':main()
