"""One saved-metadata amendment; no trainer/data/source math or task calls."""
from pathlib import Path
import hashlib
import json
import shutil

BASE=Path(__file__).resolve().parent
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('w',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def main():
    receipt_path=BASE/'training_frozen_inputs.json';draft_path=BASE/'training_clearance_draft.json'
    if sha(receipt_path)!='5cc4106b9a4ed40212e68094077966bd847f849802665b9a0533fa7861105d4f':raise ValueError('Unexpected initial freeze; no rewrite.')
    if (BASE/'fit').exists() or (BASE/'training_clearance.json').exists():raise ValueError('Cannot amend a started or approved trial.')
    preserved=BASE/'preserved_freeze_v1';preserved.mkdir(exist_ok=False)
    for path in (receipt_path,draft_path):shutil.copy2(path,preserved/path.name)
    receipt=read(receipt_path);draft=read(draft_path)
    for path in (BASE/'verify_completed_v2.py',Path(__file__),preserved/'training_frozen_inputs.json',preserved/'training_clearance_draft.json'):
        receipt['input_sha256'][path.as_posix()]=sha(path)
    receipt['input_sha256']=dict(sorted(receipt['input_sha256'].items()));receipt['input_count']=len(receipt['input_sha256'])
    receipt['owner_checker']=dict(path=(BASE/'verify_completed_v2.py').as_posix(),sha256=sha(BASE/'verify_completed_v2.py'),
        amendment='Root requested explicit evaluator role names and all_postrun_pins_exact; prior checker and unexecuted freeze preserved.')
    write(receipt_path,receipt);draft['frozen_receipt_sha256']=sha(receipt_path);write(draft_path,draft)
    print(json.dumps(dict(request_sha256=sha(BASE/'training_request.json'),frozen_receipt_sha256=sha(receipt_path),
        source_count=len(receipt['source_sha256']),input_count=receipt['input_count'],owner_checker_sha256=sha(BASE/'verify_completed_v2.py'))))
if __name__=='__main__':main()
