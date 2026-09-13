"""Create and statically verify fixed initialization before any optimizer step."""
import json
import numpy as np
import torch
import onnxruntime as ort
from fit_linear_head import CONFIG,make_actor,export
from student_linear_runtime import BASE,assert_frozen,archive,sha,FROZEN_RECEIPT


def main():
    assert_frozen();torch.set_num_threads(1);torch.manual_seed(CONFIG['seed']);np.random.seed(CONFIG['seed'])
    dest=BASE/'fit';dest.mkdir(exist_ok=False)
    data=archive(BASE/'labels/labels.npz');X=data['features'].astype(np.float32);span=data['joint_span'].astype(np.float32)
    mean=X.mean(0).astype(np.float32);std=np.maximum(X.std(0),CONFIG['feature_std_floor']).astype(np.float32)
    actor=make_actor();export(actor,mean,std,span,dest/'zero_head.onnx')
    settings=ort.SessionOptions();settings.intra_op_num_threads=1;settings.inter_op_num_threads=1
    session=ort.InferenceSession(str(dest/'zero_head.onnx'),sess_options=settings,providers=['CPUExecutionProvider'])
    output=session.run(None,{'features':X})[0]
    np.testing.assert_array_equal(output,np.zeros_like(output))
    np.testing.assert_array_equal(data['base_target']+output,data['base_target'])
    request=dict(kind='fixed_linearspan_zero_initialization_beforefit',seed=CONFIG['seed'],optimizer_steps=0,
        all1269_teacher_output_and_target_exact=True,head_sha256=sha(dest/'zero_head.onnx'),
        labels_sha256=sha(BASE/'labels/labels.npz'),source_receipt_sha256=sha(FROZEN_RECEIPT),
        dynamic_zero_parity_pending=True)
    (dest/'zero_initialization.json').write_text(json.dumps(request,indent=2)+'\n');print(json.dumps(request))


if __name__=='__main__':main()
