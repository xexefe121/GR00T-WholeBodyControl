"""Prepare only the root-selected split first layer, preserving failed v1."""
import json
import hashlib
from pathlib import Path
import shutil
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_causal_context_study_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def main():
    old=read(OLD/'training_frozen_inputs.json');source=BASE/'source_draft_v1';source.mkdir(exist_ok=False)
    for name,digest in old['source_sha256'].items():
        original=Path(old['source_directory'])/name
        if sha(original)!=digest:raise ValueError('Original source changed.')
        shutil.copyfile(original,source/name)
    path=source/'context_model.py';text=path.read_text()
    before='        return self.actor((features-self.feature_mean)/self.feature_std)'
    after='''        normalized=(features-self.feature_mean)/self.feature_std
        first=self.actor[0]
        # Keep the original1000 contraction dimensions/layout at initialization.
        # Contiguous copies are differentiable views of the same six parameters.
        original=torch.nn.functional.linear(normalized[:,:1000].contiguous(),first.weight[:,:1000].contiguous(),first.bias)
        context=torch.nn.functional.linear(normalized[:,1000:].contiguous(),first.weight[:,1000:].contiguous(),None)
        value=original+context
        for index in range(1,len(self.actor)):value=self.actor[index](value)
        return value'''
    if text.count(before)!=1:raise ValueError('Original forward differs.')
    path.write_text(text.replace(before,after),encoding='utf-8',newline='\n')
    path=source/'train_context_pair.py';text=path.read_text()
    before='changed_MatMul_dimension=True,byte_gate_required=False)'
    after="changed_MatMul_dimension=False,stored_first_layer_width=1323,first_layer_execution='split_contiguous_1000_plus_323',byte_gate_required=False)"
    if text.count(before)!=1:raise ValueError('Original drift metadata differs.')
    text=text.replace(before,after)
    before="execution_dtype='float64',public_input_dtype='float32',public_output_dtype='float32',"
    after="execution_dtype='float64',public_input_dtype='float32',public_output_dtype='float32',\n            training_first_layer_execution='split_contiguous_1000_plus_323',export_first_layer_execution='monolithic_float64_1323',"
    if text.count(before)!=1:raise ValueError('Original export metadata differs.')
    path.write_text(text.replace(before,after),encoding='utf-8',newline='\n')
    original=read(OLD/'training_request.json');proposal=dict(original)
    proposal.update(root_selected=False,source_preparation_selected=True,first_layer_execution='split_contiguous_1000_plus_323',
        export_first_layer_execution='monolithic_float64_1323',previous_attempt_root=OLD.as_posix(),
        previous_attempt_owner_sha256=sha(OLD/'owner_completion_verification_v3.json'),
        previous_attempt_counts=dict(initial_GPU32_forwards=1437,initial_GPU32_rows=367570,training_forwards=0,optimizer_updates=0),
        pending_before_actual_fit=['exact split source and synthetic review','new concrete request/budget selection','root launch clearance'])
    write(BASE/'training_request_proposal.json',proposal)
    original_map=old['source_sha256'];new_map={p.name:sha(p) for p in sorted(source.glob('*.py'))}
    write(BASE/'source_derivation.json',dict(original_source_directory=old['source_directory'],original_source_sha256=original_map,
        source_directory=source.as_posix(),source_sha256=new_map,
        changed_files=[name for name in new_map if new_map[name]!=original_map[name]],
        numerical_change='Two contiguous FP32 first-layer contractions, old bias once, sum before unchanged ELU; all parameters and losses unchanged.',
        metadata_change='Initial drift and final report disclose split training versus monolithicFP64 export.',task_model_calls=0,optimizer_updates=0,native_steps=0))
if __name__=='__main__':main()
