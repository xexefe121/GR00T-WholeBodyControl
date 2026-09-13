"""Reuse actual WSL runtime reads, replacing only reviewed source paths.

No training-corpus traversal, native object or model session is constructed.
"""
import hashlib,json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
PRIOR=NEW/'direct_target_full_state_evaluation_v1'
SOURCE=BASE/'source_draft_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    prior=read(PRIOR/'runtime_inventory.json');prep=read(BASE/'source_preparation.json');entries={}
    assert prior['recursive_training_hashes'] is False
    def add(path,reason,expected=None):
        path=Path(path).resolve();digest=sha(path)
        if expected is not None:assert digest==expected,str(path)
        entries[path.as_posix()]={'path':path.as_posix(),'sha256':digest,'bytes':path.stat().st_size,'reason':reason}
    for entry in prior['files']:
        path=Path(entry['path']).resolve()
        if not path.is_relative_to(PRIOR.resolve()):add(path,'unchanged actual runtime/native/reference input',entry['sha256'])
    current={p.relative_to(SOURCE).as_posix():sha(p) for p in SOURCE.rglob('*.py')}
    assert current==prep['source_sha256']
    for name,digest in current.items():add(SOURCE/name,'reviewed71000 response-balanced context runtime source',digest)
    fixture=NEW/'walk003_canonical_initial_fixture_v1/initial_integration_state.npz'
    baseline=NEW/'original_bfm_entry250_v1/entry250/trace.npz'
    add(fixture,'saved-only full291 baseline-to-canonical fixture check')
    with np.load(fixture,allow_pickle=False) as f,np.load(baseline,allow_pickle=False) as b:
        assert int(f['state_spec'])==8191
        a=f['state_vector'];v=b['initial_integration']
        assert a.dtype==v.dtype==np.float64 and a.shape==v.shape==(291,) and a.tobytes()==v.tobytes()
    for p in (BASE/'source_preparation.json',Path(__file__),PRIOR/'runtime_inventory.json'):
        add(p,'immutable source/runtime inventory derivation')
    forbidden=('/generation/features.npy','/generation/teacher_target.npy','/direct_target_gpu_20260911/',
               '/one_step_policy_branch_collection_resume2969_v1/collection/data/','/direct_target_full_state_secants_v1/generation/')
    assert not any(any(text in path for text in forbidden) for path in entries)
    result={'kind':'explicit_response71000_fixed_runtime_inventory','preparation_only':True,'recursive_training_hashes':False,
        'source_directory':SOURCE.as_posix(),'onnx_dependencies':prior['onnx_dependencies'],'package_allowlist':prior['package_allowlist'],
        'prior_inventory_sha256':sha(PRIOR/'runtime_inventory.json'),'source_sha256':current,
        'files':[entries[k] for k in sorted(entries)],'total_files':len(entries),'total_bytes':sum(v['bytes'] for v in entries.values()),
        'canonical_fixture':{'path':fixture.as_posix(),'sha256':sha(fixture)},
        'baseline_full291_fixture_byteexact':True,'future_final_export_and_receipts_bound':False,
        'limitations':prior['limitations'],'native_steps':0,'model_calls':0,'optimizer_updates':0}
    with (BASE/'runtime_inventory.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'inventory_sha256':sha(BASE/'runtime_inventory.json'),'files':len(entries),'MiB':result['total_bytes']/2**20}))


if __name__=='__main__':main()
