"""Derive release-only changes from reviewed55000 evaluation; zero model calls."""
import hashlib,json,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent;OLD=BASE.parent/'direct_target_continuation_evaluation_v1'
SOURCE=BASE/'source_draft_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not SOURCE.exists();SOURCE.mkdir()
    previous=json.loads((OLD/'source_derivation.json').read_text());sources={};changes=[]
    for name,entry in previous['sources'].items():
        origin=OLD/'source_draft_v1'/name;assert sha(origin)==entry['sha256']
        target=SOURCE/name;target.parent.mkdir(parents=True,exist_ok=True)
        if name=='evaluation_gate.py':
            text=origin.read_text()
            before="names=('head','checkpoint','fit_report','training_manifest','centers','query250_labels','contract')"
            after="names=('head','checkpoint','fit_report','training_manifest','training_request','centers','query250_labels','contract','source_head','normalization','export_report','export_request','export_manifest')"
            assert text.count(before)==1;text=text.replace(before,after);changes.append(dict(file=name,before=before,after=after))
            begin=text.index("    fit=read(paths['fit_report'])")
            end=text.index("    with np.load(paths['centers']",begin)
            before=text[begin:end]
            after="    from export_release_gate import validate_release\n    fit=validate_release(binding,paths,source,purpose,read=read,sha=sha,bound_file=bound_file,field=field,has_hash=has_hash)\n"
            text=text[:begin]+after+text[end:];changes.append(dict(file=name,before=before,after=after));target.write_text(text)
        else:shutil.copyfile(origin,target)
        sources[name]=dict(original=origin.as_posix(),original_sha256=entry['sha256'],sha256=sha(target),byte_identical=sha(target)==entry['sha256'])
    new=BASE/'export_release_gate.py';shutil.copyfile(new,SOURCE/new.name)
    sources[new.name]=dict(original=new.as_posix(),original_sha256=sha(new),sha256=sha(SOURCE/new.name),byte_identical=True,new_release_gate=True)
    result=dict(kind='same55000_fp64_export_evaluation_derivation',passed=True,preparation_only=True,sources=sources,
        exact_changes=changes,controller_witness_native_arithmetic_unchanged=True,original_failed_export_preserved=True,
        ordinary_final_step=55000,training_updates=0,model_calls=0,native_steps=0,
        original_source_review=dict(path=(BASE.parent/'direct_target_continuation_evaluation_source_review_v1/review.json').as_posix(),
            sha256=sha(BASE.parent/'direct_target_continuation_evaluation_source_review_v1/review.json')))
    (BASE/'source_derivation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(source_files=len(sources),unchanged_prior=29,derivation_sha256=sha(BASE/'source_derivation.json'))))
if __name__=='__main__':main()
