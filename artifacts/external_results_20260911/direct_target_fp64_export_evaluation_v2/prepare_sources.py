"""Preserve complete v1 package and derive root-audit schema alignment only."""
import hashlib,json,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent;OLD=BASE.parent/'direct_target_fp64_export_evaluation_v1'
SOURCE=BASE/'source_draft_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not SOURCE.exists();SOURCE.mkdir()
    previous=json.loads((OLD/'source_derivation.json').read_text());sources={}
    for name,entry in previous['sources'].items():
        origin=OLD/'source_draft_v1'/name;assert sha(origin)==entry['sha256']
        target=SOURCE/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(BASE/name if name=='export_release_gate.py' else origin,target)
        sources[name]=dict(original=origin.as_posix(),original_sha256=entry['sha256'],sha256=sha(target),byte_identical=sha(target)==entry['sha256'])
    result=dict(kind='same55000_fp64_export_evaluation_v2_derivation',passed=True,preparation_only=True,sources=sources,
        exact_changed_files=['export_release_gate.py'],
        parent_preparation=dict(path=(OLD/'source_preparation.json').as_posix(),sha256=sha(OLD/'source_preparation.json')),
        controller_witness_native_arithmetic_unchanged=True,original_failed_export_preserved=True,
        ordinary_final_step=55000,model_calls=0,native_steps=0,optimizer_updates=0)
    (BASE/'source_derivation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(source_files=len(sources),unchanged_prior=30,derivation_sha256=sha(BASE/'source_derivation.json'))))
if __name__=='__main__':main()
