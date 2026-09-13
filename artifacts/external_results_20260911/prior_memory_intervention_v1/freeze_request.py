"""Freeze only; no graph sessions, inference, model initialization or physics."""
import ast,hashlib,json,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent;NEW=BASE.parent;SRC=BASE/'source_snapshot_v1';RUN=NEW/'velocity_chord_student_evaluation_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
assert not (BASE/'request.json').exists()
folder=SRC/'frozen';folder.mkdir(exist_ok=False)
original=RUN/'source_snapshot_v1'
for name in ('g1_true23_bfm_seed_observations.py','g1_true23_mjbatch_bfm_seed.py','g1_true23_mpc_student.py'):
    shutil.copyfile(original/'gear_sonic/utils'/name,folder/name)
shutil.copyfile(original/'student_linear_runtime.py',folder/'student_linear_runtime.py')
binding=read(RUN/'evaluation_binding.json');inputs={entry['path']:entry['sha256'] for entry in binding['input_files']}
bundle=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1')
graphs=bundle.parent/'bfm_onnx_v2'
paths=dict(actual=RUN/'nominal/trace.npz',centers=NEW/'velocity_chord_student_v1/generation/centers.npz',
    fixed_map_report=NEW/'student_velocity_chord_fixed_map_v1/report.json',fixed_map_arrays=NEW/'student_velocity_chord_fixed_map_v1/arrays.npz',
    head=NEW/'velocity_chord_student_v1/fit/student_head.onnx',actor=graphs/'actor.onnx',backward=graphs/'backward.onnx',
    contract=bundle/'contract.json',original=bundle/'walk003/native_original.npz',original29=bundle/'walk003/original29.npz',
    motion=NEW.parent/'sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk003/reference.npz',
    onnx_dependencies=Path('E:/codex_sonic_runtime/bfm_seed_20260910/onnx_deps'))
assert sha(paths['head'])=='219b86cc4decd51671cb7aa036a944741cebed684accd91f7a03de37a7694bb5'
assert sha(paths['actual'])=='1e8e44a6c405a86138558ea89681fb225b73b2be969e2e7a608903ebb5f80872'
assert sha(paths['centers'])=='5b07595d07e262f2ad236565ec62483d600843fe27a9f4a781974dac4b0608c7'
assert sha(paths['fixed_map_report'])=='82a743ee3ce6136d4d1a265ee89505e2cfab1744e6f0348a22b4d059a04197aa'
for key,path in paths.items():
    if key!='onnx_dependencies':inputs[path.as_posix()]=sha(path)
extra=[NEW/'previous_command_design_review_v1/proposal.json',NEW/'previous_command_design_review_v1/RECOMMENDATION.md',
    RUN/'evaluation_binding.json',RUN/'nominal/report.json',RUN/'head_witness/report.json',RUN/'head_witness/witness.npz',
    NEW/'student_velocity_chord_saved_outcome_v1/report.json',NEW/'student_velocity_chord_fixed_map_v1/request.json',
    NEW/'student_velocity_chord_fixed_map_v1/diagnose_fixed_map.py',BASE/'run_durable.ps1',Path(__file__)]
for path in extra:inputs[path.as_posix()]=sha(path)
for path,digest in inputs.items():assert sha(path)==digest,path
sources={p.relative_to(SRC).as_posix():sha(p) for p in SRC.rglob('*.py')}
for path in SRC.rglob('*.py'):ast.parse(path.read_text())
write(BASE/'request.json',dict(kind='root_selected_fixed60_call_current_memory_intervention',
    source_directory=SRC.as_posix(),source_sha256=sources,input_sha256=inputs,paths={k:v.as_posix() for k,v in paths.items()},
    selected_proposal_sha256=sha(NEW/'previous_command_design_review_v1/proposal.json'),controls=list(range(250,262)),
    graph_calls=dict(backward=12,actor=24,head=24),total_graph_calls=60,
    ordering='All12 baseline backward/actor/head checks pass before any12 actor/head interventions; reuse same-state latent.',
    BFM_goals='Original native50Hz source, h8 positiongain1 yawgain2 frozen backward expressions.',
    residual_goals='v4 native reference plus original29 task features, measured heading and original offsets.',
    intervention='Replace current previous_action with matching-clock dataset2 qualified teacher prior in actor last_action and clipped previous-target/direct-prior head features; recompute BFM base and head; hold actual qpos/qvel/frame and actor lag-history fixed.',
    control250_noop_required=True,physics_steps=0,optimizer_updates=0,new_training_labels=0,new_queries=0,
    interpretation='Current-memory input intervention; not a physically coherent prior-command trajectory or replanned expert response.'))
write(BASE/'clearance_DRAFT.json',dict(approved=False,request_sha256=sha(BASE/'request.json'),launcher_sha256=sha(BASE/'run_durable.ps1'),
    total_graph_calls=60,review_path=None,review_sha256=None))
print(json.dumps(dict(request_sha256=sha(BASE/'request.json'),sources=len(sources),inputs=len(inputs),launcher_sha256=sha(BASE/'run_durable.ps1'))))
