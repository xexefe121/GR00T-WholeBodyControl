"""Future selected one-call WSL witness. Missing actual binding stops all calls."""
from pathlib import Path
import json
import sys
import traceback
import numpy as np
from evaluation_gate import require_model_ready,sha,local
from direct_runtime import native_output

BASE=Path(__file__).resolve().parent.parent


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def atomic(path,value):
    temp=Path(path).with_suffix('.tmp');write(temp,value);temp.replace(path)


def main():
    ready=require_model_ready(BASE,'witness');b=ready['binding'];binding_sha=ready['binding_sha256']
    output=BASE/'head_witness';output.mkdir(exist_ok=False)
    attempted=returned_count=0
    context={key:value.copy() for key,value in ready['center'].items()}
    context.update(span=ready['span'].copy(),default=ready['default'].copy(),limits=ready['limits'].copy())
    def count():atomic(output/'attempt.json',dict(expected_head_calls=1,attempted_head_calls=attempted,
                                                 returned_head_calls=returned_count,binding_sha256=binding_sha))
    count()
    try:
        assert sys.platform!='win32','Fixed witness requires the evaluator WSL runtime.'
        sys.path.insert(0,str(local(b['onnx_dependencies'])))
        import onnxruntime as ort
        assert ort.__version__=='1.23.2' and np.__version__=='1.26.4'
        binaries=list((Path(ort.__file__).parent/'capi').glob('*pybind11_state*.so'));assert len(binaries)==1
        runtime_sha=sha(binaries[0]);assert any(entry['sha256']==runtime_sha for entry in b['input_files'])
        settings=ort.SessionOptions();settings.intra_op_num_threads=1;settings.inter_op_num_threads=1
        settings.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        head=ort.InferenceSession(str(ready['head']),sess_options=settings,providers=['CPUExecutionProvider'])
        assert len(head.get_inputs())==1 and head.get_inputs()[0].name=='features' and head.get_inputs()[0].shape[-1]==1000
        assert len(head.get_outputs())==1 and head.get_outputs()[0].name=='normalized_target'
        features=ready['center']['features'].copy()
        attempted=1;count()
        returned=head.run(None,{'features':features[None]})
        returned_count=1
        for i,value in enumerate(returned):context['returned_'+str(i)]=np.asarray(value).copy()
        count()
        assert len(returned)==1 and returned[0].shape==(1,23) and returned[0].dtype==np.float32
        normalized=returned[0][0].copy()
        raw,target,delta=native_output(normalized,ready['default'],ready['span'],ready['limits'])
        final=require_model_ready(BASE,'witness');assert final['binding_sha256']==binding_sha
        np.savez_compressed(output/'witness.npz',features=features,normalized_target=normalized,
            raw_proposal=raw,target=target,delta=delta,span=ready['span'],default=ready['default'],limits=ready['limits'],
            control=np.asarray(250,np.int64),source_frame=np.asarray(261,np.int64),
            head_sha256=np.asarray(b['head']['sha256']),runtime_binary_sha256=np.asarray(runtime_sha))
        report=dict(kind='one_WSL_direct_absolute_target_activation_witness',pass_all=True,
            expected_head_calls=1,attempted_head_calls=attempted,returned_head_calls=returned_count,
            exact_reduced_query250_features=True,centers_row=2038,control=250,source_frame=261,
            head_sha256=b['head']['sha256'],witness_sha256=sha(output/'witness.npz'),
            binding_sha256=binding_sha,source_sha256=sha(__file__),runtime_binary_sha256=runtime_sha,
            onnxruntime_version=ort.__version__,pinned_WSL_runtime=True,intra_op_threads=1,inter_op_threads=1,
            execution_mode='ORT_SEQUENTIAL',execution_provider='CPUExecutionProvider',
            BFM_inference_calls=0,physics_steps=0,fitting_launched=False)
        write(output/'report.json',report)
    except BaseException as exc:
        np.savez_compressed(output/'failed_inputs_outputs.npz',**context)
        write(output/'failure.json',dict(pass_all=False,error=repr(exc),traceback=traceback.format_exc(),
            expected_head_calls=1,attempted_head_calls=attempted,returned_head_calls=returned_count,
            binding_sha256=binding_sha,BFM_inference_calls=0,physics_steps=0))
        raise


if __name__=='__main__':main()
