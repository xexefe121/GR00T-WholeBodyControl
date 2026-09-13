"""Finalize the parent-selected bounded diagnostic; no Torch imports/calls."""
from pathlib import Path
import ast
import hashlib
import json

BASE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def main():
    draft=BASE/'gradient_diagnostic_draft.py'
    source=BASE/'gradient_diagnostic.py'
    text=draft.read_text(encoding='utf-8').replace('Draft only: exactly three','Frozen: exactly three')
    assert text.count('weights_only=False')==1
    text=text.replace('weights_only=False','weights_only=True')
    old="        frozen()\n        state.update(stage='complete', completed=True)"
    new="        expected_counts = dict(forward_attempted=3, forward_returned=3, forward_rows_attempted=14110, forward_rows_returned=14110, gradient_attempted=3, gradient_returned=3)\n        if counters != expected_counts: raise ValueError('Fixed diagnostic call counts differ.')\n        frozen()\n        state.update(stage='complete', completed=True)"
    assert text.count(old)==1
    text=text.replace(old,new)
    ast.parse(text)
    with source.open('x',encoding='utf-8') as f:f.write(text)
    request=json.loads((BASE/'gradient_request_draft.json').read_text(encoding='utf-8'))
    request.update(prepared_only=False,root_selected=True,source_sha256=sha(source),weights_only_checkpoint_load=True)
    request['input_sha256'][(BASE/'gradient_request_draft.json').as_posix()]=sha(BASE/'gradient_request_draft.json')
    request['input_sha256'][draft.as_posix()]=sha(draft)
    for path,digest in request['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed frozen input: '+path)
    write_new(BASE/'gradient_request.json',request)
    receipt=dict(kind='bounded_gradient_source_request_freeze',source_path=source.as_posix(),source_sha256=sha(source),
        request_path=(BASE/'gradient_request.json').as_posix(),request_sha256=sha(BASE/'gradient_request.json'),
        input_sha256=request['input_sha256'],original_source_directory=request['original_source_directory'],
        budgets=request['budgets'],safe_weights_only_loader_checked=True,safe_loader_task_forwards=0,
        AST_valid=True,optimizer_updates=0,native_calls=0,executed=False)
    write_new(BASE/'gradient_frozen_inputs.json',receipt)
    print(json.dumps(dict(source_sha256=sha(source),request_sha256=sha(BASE/'gradient_request.json'),
        frozen_receipt_sha256=sha(BASE/'gradient_frozen_inputs.json'),input_pins=len(request['input_sha256']),executed=False)))

if __name__=='__main__':main()
