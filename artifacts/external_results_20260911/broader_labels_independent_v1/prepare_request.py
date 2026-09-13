"""Pin original qualifications, independent snapshots and selected label files."""
import argparse
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).parent;NEW=BASE.parent
ROOT=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
OLD=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
BUNDLE=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
COLLECTION=NEW/'pico_walk002_labels_v1/collection'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def win(p):
    s=str(p).replace('\\','/')
    if s.startswith('/mnt/'):s=s[5].upper()+':'+s[6:]
    q=Path(s);return q if q.is_absolute() else ROOT/q


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inference',action='store_true');args=parser.parse_args()
    pins={}
    def pin(p,expected=None):
        q=win(p);h=sha(q)
        assert 'walk008' not in str(q).lower()
        if expected:assert h==expected,str(q)
        s=str(q).replace('\\','/');pins[s]=h;return s
    cases=[]
    for clip,total,stop,qualname,snapname,labelsha,history,previous in (
        ('pico',6530,6230,'pico_full_root_qualification_v1','pico_all_control_snapshots_independent_v1',
         '28097c143fa1c8ac86c660a8945c83a59cf58bb9586778e5041fbeb98245a2df','fresh_seed_measured_history','fresh_seed_previous_action'),
        ('walk002',1417,1117,'walk002_hybrid_root_qualification_v1','walk002_hybrid_all_control_snapshots_independent_v1',
         'f19c08d17e2e801e169bdd8ad4018ae56efe7c2e23dc7a7e330921fff0379207','history','previous_action')):
        qp=NEW/qualname/'qualification.json';qual=read(qp)
        snap=NEW/snapname/'control_snapshots.npz';sr=read(snap.with_name('report.json'))
        case=dict(clip=clip,total_controls=total,moving_stop=stop,history_key=history,previous_key=previous,
            qualification=pin(qp),trace=pin(qual['traces']['full']['path'],qual['traces']['full']['sha256']),
            labels=pin(COLLECTION/clip/'labels.npz',labelsha),
            snapshots=pin(snap,sr['snapshots_sha256']),snapshot_report=pin(snap.with_name('report.json')),
            timeline=pin(BUNDLE/clip/'timeline.json'),original29=pin(BUNDLE/clip/'original29.npz'),
            reference=pin(OLD/'mjbatch_intent_floor_inputs_v1'/clip/'reference.npz'))
        for entry in [*qual['traces'].values(),*qual['independent_reports'].values()]:pin(entry['path'],entry['sha256'])
        for p,h in sr['hashes'].items():pin(p,h)
        pin(snap.with_name('request.json'),sr['request_sha256'])
        pin(BUNDLE/clip/'native_original.npz')
        for name in ('report.json','portable_receipt.json'):pin(OLD/'mjbatch_intent_floor_inputs_v1'/clip/name)
        cases.append(case)
    for name in ('manifest.json','contract.json','prepared_model_arrays.npz','native_prepared.xml'):pin(BUNDLE/name)
    for name,h in read(BUNDLE/'manifest.json')['meshes'].items():pin(BUNDLE/'meshes'/name,h)
    onnx=ROOT/'artifacts/teleop_six_hour_20260910/bfm_onnx_v2'
    for name in ('manifest.json','actor.onnx','backward.onnx'):pin(onnx/name)
    for path in (BASE/'source').rglob('*.py'):pin(path)
    for path in (BASE/'prepare_sources.py',BASE/'copied_original_sources.json',Path(__file__)):pin(path)
    # Runtime binary identities are independently frozen from their actual files.
    runtime=Path('E:/codex_sonic_runtime/mjbatch323_20260910/venv/lib/python3.11/site-packages')
    for folder in ('mujoco','numpy/core','scipy/spatial/transform'):
        for path in (runtime/folder).glob('*.so*'):pin(path)
    ort=Path('E:/codex_sonic_runtime/bfm_seed_20260910/onnx_deps/onnxruntime')
    pin(ort/'__init__.py')
    for path in (ort/'capi').glob('*'):
        if path.is_file() and ('.so' in path.name or path.suffix=='.py'):pin(path)
    result=dict(kind='independent_broader_actual_label_audit_request',selected_rows=6847,
        expected_graph_calls=dict(backward=6847,actor=6847),independent_inference_selected=args.inference,
        root_selection='After pure saved-array pass, one complete6847-row original BFM reevaluation; compare allraw/base/state/features exactly; no new labels, physics or fitting.',
        physics_authorized=False,fitting_authorized=False,
        cases=cases,bundle=str(BUNDLE).replace('\\','/'),onnx=str(onnx).replace('\\','/'),
        onnx_dependencies='E:/codex_sonic_runtime/bfm_seed_20260910/onnx_deps',
        saved_normalization=pin(COLLECTION/'existing_normalization.npz'),
        original_normalization=pin(NEW/'fast_controller_phase_fit_v1/fit/teacher_fit.npz'),
        preserved_exit_receipt=pin(NEW/'pico_walk002_labels_v1/process/exit.json'))
    if args.inference:
        report=BASE/'saved_array_audit/report.json';r=read(report)
        assert r['pass_all'] and r['selected_rows_checked']==6847
        assert r['actor_inference_calls']==r['backward_inference_calls']==r['physics_steps']==0
        result['saved_audit_report']=pin(report)
        pin(BASE/'saved_array_audit/comparison_log.npz')
        pin(BASE/'saved_request.json')
        pin(BASE/'focused.xml')
    result['input_hashes']=pins
    path=BASE/('inference_request.json' if args.inference else 'saved_request.json')
    assert not path.exists(),'Preserve the existing audit request.'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(request=str(path),sha256=sha(path),pins=len(pins))))


if __name__=='__main__':main()
