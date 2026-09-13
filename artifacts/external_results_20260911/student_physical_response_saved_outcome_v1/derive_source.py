import ast,hashlib,json
from pathlib import Path
N=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911');O=Path(__file__).parent
old=N/'student_velocity_chord_saved_outcome_v1/audit_saved.py'
source=old.read_text();changes=[]
def change(a,b):
    global source
    assert source.count(a)==1,(a,source.count(a));source=source.replace(a,b)
    changes.append(dict(old=a,new=b))
change("RUN=NEW/'velocity_chord_student_evaluation_v1'", "RUN=NEW/'one_step_physical_student_evaluation_v1'")
change("FIT=NEW/'velocity_chord_student_v1'", "FIT=NEW/'one_step_physical_student_v1'\nGEN=NEW/'velocity_chord_student_v1'")
change("centers=FIT/'generation/centers.npz',probes=FIT/'generation/features.npy',", "centers=GEN/'generation/centers.npz',probes=GEN/'generation/features.npy',\n        physical_manifest=NEW/'one_step_policy_branch_collection_resume2969_v1/collection/data/manifest.json',\n        physical_features=NEW/'one_step_policy_branch_collection_resume2969_v1/collection/data/endpoint_features.npy',\n        physical_valid=NEW/'one_step_policy_branch_collection_resume2969_v1/collection/data/label_valid.npy',\n        root_physics=NEW/'student_physical_response_independent_physics_v1/report.json',\n        root_intent=NEW/'student_physical_response_independent_intent_v1/report.json',")
change("final_export_review=NEW/'velocity_chord_final_export_review_v1/review.json',", "final_export_review=NEW/'one_step_physical_final_fit_review_v1/review.json',")
change("root_fit_review=NEW/'velocity_fit_evidence_independent_v1/report.json',", "root_fit_review=NEW/'physical_fit_evidence_independent_v1/report.json',")
change("driver=SRC/'evaluate_velocity_chord_student.py',", "driver=SRC/'evaluate_physical_response_student.py',")
change("assert sha(p['nominal_trace'])==report['trace_sha256']", "assert sha(p['nominal_trace'])==report['trace_sha256']=='38428ae27055be7c2e771c5a23056c858f004ef38f260a0b45a57662d8d0aae3'\n    assert sha(p['root_physics'])=='ede8ff939da69885f509785e62e6c6cc8d0ecb190b2cb8fb15eb1e5ec4394133'\n    physics=read(p['root_physics']);assert physics['compared_physics_steps']==3193\n    assert all(physics['original_trace_comparison'].values())\n    manifest=read(p['physical_manifest'])\n    assert manifest['complete'] is True and manifest['labels_valid']==3054\n    for key,path in [('endpoint_features',p['physical_features']),('label_valid',p['physical_valid'])]:\n        assert sha(path)==manifest['arrays'][key]['sha256']")
change("kind='one_saved_final70000_outcome_audit'", "kind='one_saved_final75000_outcome_audit'")
change("augmented_lo=np.minimum(nominal_lo,probes.min(0));augmented_hi=np.maximum(nominal_hi,probes.max(0))", "augmented_lo=np.minimum(nominal_lo,probes.min(0));augmented_hi=np.maximum(nominal_hi,probes.max(0))\n        physical_features=np.load(p['physical_features'],mmap_mode='r');physical_valid=np.load(p['physical_valid'],mmap_mode='r')\n        assert physical_features.shape==(3054,1069) and physical_valid.shape==(3054,) and physical_valid.all()\n        augmented_lo=np.minimum(augmented_lo,physical_features[physical_valid].min(0))\n        augmented_hi=np.maximum(augmented_hi,physical_features[physical_valid].max(0))")
change("saved_fit_at_query250_expert_input_rmse_rad=rms(matched_target[i]-labels['expert_target'][qindices[i]]),", "saved_fit_at_query250_expert_input_rmse_rad=rms(matched_target[i]-labels['expert_target'][qindices[i]]),\n                actual_vs_saved_query250_base_rms_rad=rms(row['base_target']-labels['base_target'][qindices[i]]),\n                actual_vs_saved_query250_head_rms_rad=rms(row['delta']-matching_delta[i]),\n                actual_vs_saved_query250_unclipped_proposal_rms_rad=rms(row['raw_proposal']-(labels['base_target'][qindices[i]]+matching_delta[i])),")
change("speed_ratio=float(ratios[j]),qpos=float(a['physics_qpos'][-1,7+j])))", "speed_ratio=float(ratios[j]),qpos=float(a['physics_qpos'][-1,7+j]),\n                maximum_range_joint=names[int(np.maximum(limits[:,0]-a['physics_qpos'][-1,7:],a['physics_qpos'][-1,7:]-limits[:,1]).argmax())],\n                maximum_range_excess_rad=float(np.maximum(np.maximum(limits[:,0]-a['physics_qpos'][-1,7:],a['physics_qpos'][-1,7:]-limits[:,1]),0).max())))")
change("training_envelope='Both3057 nominal centers and full143679 nominal+probe feature universe; coordinate ranges are descriptive, not a feasibility or stability gate.'", "training_envelope='All3057 nominal centers,140622 velocity probes and3054 valid physical endpoint features; coordinate ranges are descriptive, not a feasibility or stability gate.'")
change("comparison_limitation='Same-clock targets and saved head predictions belong to distinct expert trajectories after control250; no expert counterfactual evaluated at actual student states.'", "comparison_limitation='Same-clock targets and saved Windows Torch head predictions belong to distinct expert trajectories after control250; comparisons are descriptive. The WSL first activation has its own exact witness. No expert counterfactual evaluated by this audit.'")
change("write(BASE/'checks.json',CHECKS)", "write(BASE/'checks.json',CHECKS)\n    assert sha(__file__)==request['source_sha256']\n    for name,digest in request['input_sha256'].items():assert sha(local(name))==digest,name")
ast.parse(source)
out=O/'audit_saved.py'
with out.open('x',encoding='utf-8') as f:f.write(source)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with (O/'source_derivation.json').open('x',encoding='utf-8') as f:
    json.dump(dict(original_path=old.as_posix(),original_sha256=sha(old),derived_sha256=sha(out),changes=changes,
        observation_goal_history_action_arithmetic_unchanged=True,physical_envelope_extension=True,
        no_model_or_native_calls=True),f,indent=2);f.write('\n')
print(json.dumps(dict(source_sha256=sha(out),changes=len(changes))))
