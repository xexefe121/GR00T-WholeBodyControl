import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((BASE/'source_check.json').read_text())
parse=json.loads((BASE/'powershell_parse.json').read_text(encoding='utf-8-sig'))
assert len(parse)==4 and all(x['errors']==[] for x in parse)
pins=r['source_sha256']
for path,digest in pins.items():assert sha(path)==digest,path
for name in ('source_check.json','powershell_parse.json','finalize.py'):pins[(BASE/name).as_posix()]=sha(BASE/name)
review=dict(verdict='CLEAR',source_review_pass=True,preparation_only=True,source_sha256=pins,
    existing_synthetic_tests=21,generated_templates_parsed=4,
    findings_resolved=['WSL initial directory is /, so the pinned bootstrap can mount E before absolute script execution.',
        'Root saved-fit audit subject gate binds actual fit report, head and checkpoint.'],
    reviewed_contract=['Only actual completed fit/export and literal six-dataset subjects can be bound.',
        'Fresh one-call WSL witness is separate from canonical 1569 plus conditional250.',
        'Same reviewed 30-source direct adapter; no learned BFM calls or new math.',
        'CreateNew execution lock, final concrete reviewer binding, hidden captured-handle wait and immutable raw exit.',
        'Diagnostic noncompletion/native/quiet/parity failures produce nonzero exit even when Python returns zero.',
        'All launch inputs rehashed before and after child execution; no automatic retry.'],
    execution_clearance=False,task_model_calls=0,native_steps=0,optimizer_updates=0,
    limitation='Concrete witness and canonical binding/launcher receipts need their own final reviews.')
with (BASE/'review.json').open('x') as f:json.dump(review,f,indent=2);f.write('\n')
print(sha(BASE/'review.json'))
