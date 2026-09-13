"""Preserve unexecuted v1 and bind saved-only completion checker v2."""
from pathlib import Path
import json
import shutil
from prepare_execution import BASE,read,write,sha
def main():
    if (BASE/'fit').exists() or (BASE/'training_clearance.json').exists():raise ValueError('Only unexecuted amendment permitted.')
    preserved=BASE/'unexecuted_freeze_v1';preserved.mkdir(exist_ok=False)
    for name in ('training_frozen_inputs.json','training_clearance_draft.json'):
        shutil.copyfile(BASE/name,preserved/name)
    receipt=read(BASE/'training_frozen_inputs.json')
    for name in ('verify_completed_v2.py','test_owner_v2.py','owner_v2_tests.xml','amend_owner_freeze_v2.py'):
        receipt['input_sha256'][(BASE/name).as_posix()]=sha(BASE/name)
    receipt['owner_checker']='verify_completed_v2.py'
    receipt['preserved_unexecuted_v1_sha256']=sha(preserved/'training_frozen_inputs.json')
    for p,digest in receipt['input_sha256'].items():
        if sha(p)!=digest:raise ValueError('Changed pin: '+p)
    with (BASE/'training_frozen_inputs.json').open('w',encoding='utf-8') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    clear=read(BASE/'training_clearance_draft.json');clear['frozen_receipt_sha256']=sha(BASE/'training_frozen_inputs.json')
    with (BASE/'training_clearance_draft.json').open('w',encoding='utf-8') as f:json.dump(clear,f,indent=2,allow_nan=False);f.write('\n')
    result=dict(request_sha256=sha(BASE/'training_request.json'),frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),
        launcher_sha256=sha(BASE/'run_fit_durable_v1.ps1'),owner_checker_sha256=sha(BASE/'verify_completed_v2.py'),
        inputs=len(receipt['input_sha256']),sources=len(receipt['source_sha256']),task_model_calls=0)
    write(BASE/'owner_freeze_amendment_v2.json',result);print(json.dumps(result))
if __name__=='__main__':main()
