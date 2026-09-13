"""Bind the now-completed root repair review; preserve pre-review metadata."""
from pathlib import Path
import hashlib,json,shutil
BASE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
if (BASE/'fit').exists() or (BASE/'training_clearance.json').exists() or (BASE/'fit_process_v1').exists():raise ValueError('Cannot amend a launched/cleared experiment.')
review=BASE.parent/'direct_target_response_repair_root_review_v1/review.json'
if sha(review)!='935b27c9f93fa21f31552dd0707174076f490d26a18f00dfd28eb9eb3ff03280':raise ValueError('Actual root repair review identity differs.')
record=read(review)
if record['data_export_source_review_pass'] is not True or record['repair_review_pass'] is not True:raise ValueError('Actual root repair review failed.')
preserved=BASE/'metadata_before_repair_review';preserved.mkdir(exist_ok=False)
for name in ('training_request.json','training_frozen_inputs.json','training_clearance_draft.json'):shutil.copyfile(BASE/name,preserved/name)
request=read(BASE/'training_request.json');request['subjects']['root_repair_source_review']=dict(path=review.as_posix(),sha256=sha(review),
    pass_field='data_export_source_review_pass',required_fields=dict(repair_review_pass=True,source_review_pass=True))
write(BASE/'training_request.json',request)
receipt=read(BASE/'training_frozen_inputs.json');receipt['training_request_sha256']=sha(BASE/'training_request.json')
receipt['input_sha256'][review.as_posix()]=sha(review);receipt['input_sha256'][Path(__file__).as_posix()]=sha(__file__)
for path,digest in receipt['input_sha256'].items():
    if sha(path)!=digest:raise ValueError('Frozen input changed: '+path)
for name,digest in receipt['source_sha256'].items():
    if sha(Path(receipt['source_directory'])/name)!=digest:raise ValueError('Frozen source changed.')
write(BASE/'training_frozen_inputs.json',receipt)
clear=read(BASE/'training_clearance_draft.json');clear['request_sha256']=sha(BASE/'training_request.json');clear['frozen_receipt_sha256']=sha(BASE/'training_frozen_inputs.json')
write(BASE/'training_clearance_draft.json',clear)
print(json.dumps(dict(request_sha256=clear['request_sha256'],frozen_receipt_sha256=clear['frozen_receipt_sha256'],
    launcher_sha256=clear['launcher_sha256'],inputs=len(receipt['input_sha256']),sources=len(receipt['source_sha256']),model_calls=0)))
