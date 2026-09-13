"""Pure direct-feature reproduction of qualified saved nominal inputs."""
from pathlib import Path
import hashlib
import json
import sys
import traceback
import numpy as np

BASE=Path(__file__).resolve().parent;NEW=BASE.parent;SOURCE=BASE/'source_draft_v1'
sys.path.insert(0,str(SOURCE))
from direct_features import DirectFeatures


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def archive(path):
    with np.load(path,allow_pickle=False) as z:return {key:z[key].copy() for key in z.files}
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)

out=BASE/'saved_feature_checks_v1';out.mkdir(exist_ok=False)
pins={}
def pin(path,expected=None):
    path=Path(path).resolve();actual=sha(path)
    if expected is not None:assert actual==expected,str(path)
    pins[str(path)]=actual
    return path
try:
    for path in (Path(__file__),SOURCE/'direct_features.py',SOURCE/'gear_sonic/utils/g1_true23_mpc_student.py',BASE/'feature_derivation.json'):pin(path)
    contract=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json')
    c=json.loads(pin(contract,'d1641ce4f2e9016f5008e5d4be3cc24ded46d7f9daf9cfddaab84095f98eb99b').read_text())
    bundle=contract.parent
    centers=archive(pin(NEW/'velocity_chord_student_v1/generation/centers.npz','5b07595d07e262f2ad236565ec62483d600843fe27a9f4a781974dac4b0608c7'))
    prior=json.loads(pin(NEW/'one_step_physical_student_evaluation_v1/evaluation_binding.json').read_text())
    prior_pins={str(Path(e['path']).resolve()):e['sha256'] for e in prior['input_files']}
    ref=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk003/reference.npz')
    original=bundle/'walk003/original29.npz'
    cases=[dict(name='walk003_three_nominal',features=centers['features'],qpos=centers['qpos'],qvel=centers['qvel'],
        frames=centers['source_frame'],controls=centers['control'],reference=pin(ref,prior_pins[str(ref.resolve())]),
        original=pin(original,prior_pins[str(original.resolve())]))]
    broader_path=NEW/'broader_labels_independent_v1/saved_request.json'
    broader=json.loads(pin(broader_path).read_text())
    broader_pins={str(Path(k).resolve()):v for k,v in broader['input_hashes'].items()}
    for item in broader['cases']:
        z=archive(pin(item['labels'],broader_pins[str(Path(item['labels']).resolve())]))
        cases.append(dict(name=item['clip'],features=z['features'],qpos=z['teacher_qpos'],qvel=z['teacher_qvel'],
                          frames=z['source_frame'],controls=z['control'],
                          reference=pin(item['reference'],broader_pins[str(Path(item['reference']).resolve())]),
                          original=pin(item['original29'],broader_pins[str(Path(item['original29']).resolve())])))
    write(out/'request.json',dict(input_sha256=pins,requested_rows=9904,models=0,native_steps=0,
                                target='original features[0:52] + features[75:1023]',first_query_center=2038))
    results=[];first=None
    for case in cases:
        features=DirectFeatures(archive(case['reference']),archive(case['original']),c)
        digest=hashlib.sha256();count=len(case['features'])
        assert case['features'].shape==(count,1069) and case['features'].dtype==np.float32
        assert np.array_equal(case['frames'],case['controls']+11)
        for row in range(count):
            expected=case['features'][row,np.r_[0:52,75:1023]].copy()
            actual=features(case['qpos'][row],case['qvel'][row],int(case['frames'][row]))
            if not exact(actual,expected):
                np.savez_compressed(out/'first_mismatch.npz',actual=actual,expected=expected,qpos=case['qpos'][row],qvel=case['qvel'][row],frame=case['frames'][row],row=row,case=case['name'])
                raise ValueError('direct feature mismatch '+case['name']+' row '+str(row))
            digest.update(actual.tobytes())
            if case['name']=='walk003_three_nominal' and row==2038:
                assert case['controls'][row]==250 and case['frames'][row]==261
                first=dict(row=2038,control=250,source_frame=261,shape=list(actual.shape),dtype=str(actual.dtype),sha256=hashlib.sha256(actual.tobytes()).hexdigest(),byte_exact=True)
        results.append(dict(case=case['name'],rows=count,all_kept_feature_bytes_exact=True,rolling_sha256=digest.hexdigest()))
        print(results[-1],flush=True)
    assert sum(item['rows'] for item in results)==9904
    for path,value in pins.items():assert sha(path)==value,path
    assert not any(name in sys.modules for name in ('mujoco','onnxruntime','torch'))
    write(out/'report.json',dict(passed=True,rows=9904,cases=results,first_query250=first,input_sha256=pins,
        request_sha256=sha(out/'request.json'),all_final_pins_exact=True,model_calls=0,native_steps=0,optimizer_calls=0,
        runtime_numpy=np.__version__,no_controller_installed=True))
except BaseException as exc:
    write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),input_pins=pins,model_calls=0,native_steps=0))
    raise
