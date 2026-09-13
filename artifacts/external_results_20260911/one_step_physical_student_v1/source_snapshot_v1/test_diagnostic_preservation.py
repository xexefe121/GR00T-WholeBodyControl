"""Pure stub fault evidence; no real network, optimizer, BFM or dynamics."""
from pathlib import Path
import tempfile
import json
import hashlib
import numpy as np
import torch
from chord_fit_diagnostics import evaluate_fixed

class FailingStub:
    def __init__(self):self.calls=0
    def __call__(self,x):
        self.calls+=1
        if self.calls==2:raise RuntimeError('fixed stub probe fault')
        return torch.zeros((len(x),23))

def main():
    torch.set_num_threads(1)
    ledger=dict(diagnostic_torch_rows_attempted=0,diagnostic_torch_rows_returned=0,
        head_onnx_calls_attempted=0,head_onnx_calls_returned=0)
    with tempfile.TemporaryDirectory() as folder:
        dest=Path(folder);zero=np.zeros(1069,np.float32)
        center=np.broadcast_to(zero,(3057,1069));probes=np.broadcast_to(zero,(3057,23,2,1069))
        try:
            evaluate_fixed(FailingStub(),center,probes,np.zeros((7,1069),np.float32),zero,np.ones(1069,np.float32),
                np.ones(23,np.float32),dest/'unused.onnx',ledger,dest,'stub')
            raise AssertionError('Stub fault not raised')
        except RuntimeError as exc:assert str(exc)=='fixed stub probe fault'
        assert ledger['diagnostic_torch_rows_attempted']==3313 and ledger['diagnostic_torch_rows_returned']==3057
        assert ledger['head_onnx_calls_attempted']==ledger['head_onnx_calls_returned']==0
        with np.load(dest/'stub_failed_diagnostic_arrays.npz') as saved:
            assert int(saved['torch_rows_returned'])==3057 and int(saved['onnx_rows_returned'])==0
            assert np.all(saved['predicted_delta'][:3057]==0) and np.isnan(saved['predicted_delta'][3057:]).all()
            assert np.isnan(saved['onnx_predicted_delta']).all()
    source=Path(__file__).parent;out=source.parent/'diagnostic_preservation_pure_tests.json';assert not out.exists()
    result=dict(passed=True,partial_returned_rows_preserved=True,explicit_uncomputed_nan_suffix=True,
        exact_attempted_vs_returned_row_counters=True,no_unbudgeted_ORT_call_on_torch_failure=True,
        actual_model_evaluations=0,BFM_calls=0,optimizer_updates=0,physics_steps=0,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),source/'chord_fit_diagnostics.py']})
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print('PURE DIAGNOSTIC PRESERVATION PASS')

if __name__=='__main__':main()
