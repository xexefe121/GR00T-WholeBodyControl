"""Pure source derivation; no Torch/checkpoint/ONNX invocation."""
from pathlib import Path
import difflib,hashlib,json
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
ORIGINAL=NEW/'direct_target_causal_response_balanced_student_v2/source_snapshot_v1/response_promoted.py'
SOURCE=BASE/'source_draft_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    SOURCE.mkdir(exist_ok=False)
    old=ORIGINAL.read_text()
    changes=[("saved['ordinary_final_step']!=71000","saved['ordinary_final_step']!=81000"),
        ('Exact warm ordinary71000 subject required','Exact proposed ordinary81000 subject required'),
        ('[(256,1323),(256,256),(23,256)]','[(512,1323),(512,512),(23,512)]'),
        ('same_71000_context_weights_fp64_execution','same_81000_width512_context_weights_fp64_execution')]
    text=old
    for a,b in changes:
        assert text.count(a)==1,(a,text.count(a));text=text.replace(a,b)
    target=SOURCE/'width512_promoted.py'
    with target.open('x',encoding='utf-8') as f:f.write(text)
    with (BASE/'source_changes.diff').open('x') as f:
        f.write(''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile=str(ORIGINAL),tofile=str(target))))
    with (BASE/'derivation.json').open('x') as f:
        json.dump(dict(original={'path':ORIGINAL.as_posix(),'sha256':sha(ORIGINAL)},derived={'path':target.as_posix(),'sha256':sha(target)},
            exact_replacements=changes,actual_checkpoint_loaded=False,actual_export_created=False),f,indent=2)

if __name__=='__main__':main()
