"""Preserve unexecuted metadata; name the root review's actual positive field."""
from pathlib import Path
import json,hashlib,shutil
BASE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
if (BASE/'fit').exists() or (BASE/'training_clearance.json').exists() or (BASE/'fit_process_v1').exists():
    raise ValueError('Metadata amendment allowed only before any fit/clearance/dispatch.')
preserve=BASE/'metadata_preserved_v1';preserve.mkdir(exist_ok=False)
for name in ('training_request.json','training_frozen_inputs.json','training_clearance_draft.json','prepare_execution.py'):
    shutil.copyfile(BASE/name,preserve/name)
request=read(BASE/'training_request.json');subject=request['subjects']['root_source_data_export_review'];review=read(subject['path'])
if subject['pass_field']!='passed' or review['data_export_source_review_pass'] is not True:
    raise ValueError('Expected actual root review field absent.')
subject['pass_field']='data_export_source_review_pass'
for key,value in request['subjects'].items():
    if 'pass_field' in value:
        record=read(value['path'])
        if record[value['pass_field']] is not True:raise ValueError('Required qualification failed: '+key)
        for name,expected in value.get('required_fields',{}).items():
            if record.get(name)!=expected:raise ValueError('Required qualification value differs: '+key+'/'+name)
source=BASE/'prepare_execution.py';text=source.read_text(encoding='utf-8')
old="        pass_field='passed')\n    write(BASE/'training_request.json',proposal)"
new="        pass_field='data_export_source_review_pass')\n    write(BASE/'training_request.json',proposal)"
if text.count(old)!=1:raise ValueError('Expected generator field not unique.')
source.write_text(text.replace(old,new),encoding='utf-8',newline='\n')
write(BASE/'training_request.json',request)
receipt=read(BASE/'training_frozen_inputs.json');receipt['training_request_sha256']=sha(BASE/'training_request.json')
receipt['input_sha256'][source.as_posix()]=sha(source)
receipt['input_sha256'][Path(__file__).as_posix()]=sha(__file__)
for path,digest in receipt['input_sha256'].items():
    if sha(path)!=digest:raise ValueError('Changed input during metadata amendment: '+path)
for name,digest in receipt['source_sha256'].items():
    if sha(Path(receipt['source_directory'])/name)!=digest:raise ValueError('Source changed.')
write(BASE/'training_frozen_inputs.json',receipt)
clear=read(BASE/'training_clearance_draft.json');clear['request_sha256']=sha(BASE/'training_request.json');clear['frozen_receipt_sha256']=sha(BASE/'training_frozen_inputs.json')
write(BASE/'training_clearance_draft.json',clear)
note=dict(metadata_only=True,all25_model_sources_unchanged=True,actual_model_calls=0,optimizer_updates=0,native_steps=0,
    reason='Root review uses data_export_source_review_pass, not passed; fix request role before any execution.',
    preserved_subjects={p.name:sha(p) for p in preserve.iterdir()},request_sha256=clear['request_sha256'],
    frozen_receipt_sha256=clear['frozen_receipt_sha256'],launcher_sha256=clear['launcher_sha256'],
    input_count=len(receipt['input_sha256']),source_count=len(receipt['source_sha256']))
write(BASE/'metadata_amendment_v1.json',note);print(json.dumps(note))
