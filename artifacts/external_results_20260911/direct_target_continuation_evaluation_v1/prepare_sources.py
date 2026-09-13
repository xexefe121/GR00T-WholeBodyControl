"""Prepare final55000 evaluation sources without model/native calls or bindings."""
import hashlib,json,shutil
from pathlib import Path

BASE=Path(__file__).resolve().parent;OLD=BASE.parent/'direct_target_student_evaluation_v1'
SOURCE=BASE/'source_draft_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not SOURCE.exists();SOURCE.mkdir()
    previous=json.loads((OLD/'source_preparation.json').read_text())
    sources={};changes=[]
    for name,digest in previous['source_sha256'].items():
        origin=OLD/'source_draft_v1'/name;assert sha(origin)==digest
        target=SOURCE/name;target.parent.mkdir(parents=True,exist_ok=True)
        if name=='evaluation_gate.py':
            text=origin.read_text()
            replacements=[("assert binding['ordinary_final_step']==5000","assert binding['ordinary_final_step']==55000"),
                ("assert fit['ordinary_final_step']==5000 and fit['additional_updates']==5000",
                 "assert fit['ordinary_final_step']==55000 and fit['additional_updates']==50000")]
            for before,after in replacements:
                assert text.count(before)==1;text=text.replace(before,after);changes.append(dict(file=name,before=before,after=after))
            target.write_text(text)
        else:shutil.copyfile(origin,target)
        sources[name]=dict(original=origin.as_posix(),original_sha256=digest,sha256=sha(target),byte_identical=sha(target)==digest)
    result=dict(kind='direct55000_runtime_source_derivation',passed=True,preparation_only=True,
        ordinary_final_step=55000,additional_updates=50000,sources=sources,exact_changes=changes,
        strict_oracle_controller_and_history_arithmetic_unchanged=True,
        learned_BFM_calls=0,requested_main_controls=1569,conditional_hold_controls=250,
        new_head_binding_created=False,model_calls=0,native_steps=0,
        original_source_review=dict(path=(BASE.parent/'direct_target_evaluation_source_review_v1/review.json').as_posix(),
            sha256=sha(BASE.parent/'direct_target_evaluation_source_review_v1/review.json')))
    (BASE/'source_derivation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(source_files=len(sources),unchanged=sum(v['byte_identical'] for v in sources.values()),
        derivation_sha256=sha(BASE/'source_derivation.json'))))

if __name__=='__main__':main()
