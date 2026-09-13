"""Saved-array/source preflight and explicit native-only inventory; no native calls."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
sys.path.insert(0,str(BASE/'source_draft_v1'))
from replay_core import Segment,archive,validate_segment,exact

ROOT=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
BUNDLE=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
VENV=Path('E:/codex_sonic_runtime/mjbatch323_20260910/venv')
SITE=VENV/'lib/python3.11/site-packages'
PACKAGES=('mujoco','mujoco.libs','numpy','numpy.libs','absl','etils','fsspec','glfw','OpenGL','packaging','zipp')
TRACES={
    'expert_main':NEW/'bfm_entry250_actual_oracle_v1/nominal/trace.npz',
    'expert_hold':NEW/'bfm_entry250_actual_oracle_v1/post_lifecycle_hold_5s/trace.npz',
    'direct_failure':NEW/'direct_target_student_evaluation_v1/nominal/trace.npz'}
REPORTS={
    'expert_main':NEW/'bfm250_expert_full_independent_physics_v1/report.json',
    'expert_hold':NEW/'bfm250_expert_hold_independent_physics_v1/report.json',
    'direct_failure':NEW/'direct_target_independent_physics_v1/report.json'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def has_hash(v,d):
    if isinstance(v,dict):return any(has_hash(x,d) for x in v.values())
    if isinstance(v,list):return any(has_hash(x,d) for x in v)
    return v==d

def main():
    assert not (BASE/'proposal.json').exists()
    pins={}
    def add(p,reason):
        p=Path(p).resolve();assert p.is_file(),str(p)
        pins[p.as_posix()]=dict(path=p.as_posix(),sha256=sha(p),bytes=p.stat().st_size,reason=reason)
    arrays={name:archive(path) for name,path in TRACES.items()}
    segments=[Segment('expert_main',arrays['expert_main'],0,1569,1569,15690),
        Segment('expert_hold',arrays['expert_hold'],1569,250,250,2500),
        Segment('direct_failure',arrays['direct_failure'],0,316,1569,3158,(315,8,'native_joint_bound'))]
    schema={}
    for segment in segments:
        aliases=validate_segment(segment);schema[segment.name]=dict(aliases=aliases,
            fields={key:dict(shape=list(value.shape),dtype=value.dtype.str) for key,value in segment.arrays.items()
                    if key.startswith('physics_') or key in ('target','global_control','control_integration_before','initial_integration','final_integration')})
        assert not np.any(segment.arrays[aliases['warnings']]) and not np.any(segment.arrays['physics_warning_lastinfo'])
        report=read(REPORTS[segment.name])
        assert report['recorded_trace_reproduced_through_last_sample'] is True
        assert report['physics_steps']==segment.recorded_steps
        assert report['feasible'] is (segment.expected_issue is None)
        comparisons=report['original_trace_comparison']
        assert len(comparisons)==7 and all(v is True for v in comparisons.values())
        assert has_hash(report,sha(TRACES[segment.name]))
        add(TRACES[segment.name],'actually replayed original saved native commands and measurements')
        add(REPORTS[segment.name],'independent original native parity qualification')
    fixture_path=NEW/'walk003_canonical_initial_fixture_v1/initial_integration_state.npz'
    fixture=archive(fixture_path);assert int(fixture['state_spec'])==8191 and fixture['state_vector'].dtype==np.float64
    assert fixture['state_vector'].shape==(291,)
    for name in ('expert_main','direct_failure'):assert exact(arrays[name]['initial_integration'],fixture['state_vector'])
    assert exact(arrays['expert_main']['final_integration'],arrays['expert_hold']['initial_integration'])
    for key in ('physics_qpos','physics_qvel','physics_time','physics_expected_time','physics_warning_lastinfo','physics_warning_number'):
        assert exact(arrays['expert_main'][key][-1],arrays['expert_hold'][key][0]),key
    add(fixture_path,'canonical full291 initialization, never q/dq reconstruction')
    for p in (BASE/'source_draft_v1').rglob('*.py'):add(p,'artifact-local source only')
    for p in (BASE/'prepare_sources.py',BASE/'source_derivation.json',Path(__file__),BASE/'stub_tests_v3.xml'):
        add(p,'source and synthetic evidence')
    for name in ('manifest.json','native_prepared.xml','prepared_model_arrays.npz','contract.json','walk003/native_original.npz'):
        add(BUNDLE/name,'actual model-loader read')
    for name,digest in read(BUNDLE/'manifest.json')['meshes'].items():
        p=BUNDLE/'meshes'/name;assert sha(p)==digest;add(p,'actual native geometry and model-loader hash')
    for name in PACKAGES:
        directory=SITE/name;assert directory.is_dir()
        for p in directory.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and (p.suffix in ('.py','.json','.pth') or '.so' in p.name):
                add(p,'conservative code closure for actual native/NumPy package '+name)
    for name in ('typing_extensions.py','_virtualenv.py','_virtualenv.pth'):
        if (SITE/name).is_file():add(SITE/name,'existing runtime import initialization')
    add(VENV/'pyvenv.cfg','WSL runtime configuration')
    add(ROOT/'artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh','fixed runtime bootstrap')
    for rel in ('independent_native_stepper_root_review_v2/review_v3.json',
                'bfm250_expert_root_qualification_v1/qualification.json',
                'bfm250_expert_full_independent_intent_v1/report.json',
                'bfm250_expert_hold_independent_intent_v1/report.json',
                'direct_target_student_evaluation_v1/nominal/report.json',
                'direct_target_student_evaluation_v1/nominal/strict_failure_state.npz',
                'direct_target_student_evaluation_v1/evaluation_completion_verification.json'):
        add(NEW/rel,'immutable source/original-run qualification and expected failure evidence')
    assert not any('/onnx' in path.lower() or 'torch' in path.lower() or '/generation/' in path.lower() for path in pins)
    proposal=dict(kind='native_stepper_equivalence_source_proposal',source_preparation_only=True,execution_selected=False,
        source_directory=(BASE/'source_draft_v1').as_posix(),native_bundle=BUNDLE.as_posix(),canonical_fixture=fixture_path.as_posix(),
        traces={name:p.as_posix() for name,p in TRACES.items()},independent_reports={name:p.as_posix() for name,p in REPORTS.items()},
        planned_stages=[dict(stage='witness',native_step_budget=0,serialization_budget=2,output='witness'),
            dict(stage='replay',native_step_budget=21348,serialization_budget=8,output='replay')],
        expected_results=dict(expert_attempted_returned_captured_verified=18190,direct_attempted_returned_captured=3158,
            direct_verified=3157,direct_issue=[315,8,'native_joint_bound'],total_serializations=10),
        saved_schema=schema,input_files=list(pins.values()),input_bytes=sum(e['bytes'] for e in pins.values()),
        actual_expected_MJB_not_yet_produced=True,hold_has_no_restore=True,
        final_review_contract='request_subject object must contain the exact actual request absolute path and SHA256',
        model_inference_calls=0,optimizer_updates=0,other_oracle_native_steps=0,plant_foundation_connected=False,
        real_wallclock_experiment=False,hardware_authorized=False)
    (BASE/'proposal.json').write_text(json.dumps(proposal,indent=2)+'\n')
    print(json.dumps(dict(proposal_sha256=sha(BASE/'proposal.json'),pins=len(pins),MiB=proposal['input_bytes']/2**20,
        saved_schemas_validated=3,model_calls=0,native_steps=0)))

if __name__=='__main__':main()
