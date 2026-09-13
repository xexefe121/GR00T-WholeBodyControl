"""Freeze collector inputs only after exact root full-trajectory qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

BASE=Path(__file__).resolve().parent;TASK=BASE.parent
RUN=TASK/'bfm_entry250_actual_oracle_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def windows(path):
    value=str(path)
    if value.startswith('/mnt/') and len(value)>7:value=value[5].upper()+':/'+value[7:]
    return Path(value)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qualification',type=Path,required=True)
    args=parser.parse_args();decision=read(args.qualification)
    assert decision['root_authorized_extraction'] is True and decision['model_fitting_authorized'] is False
    assert sha(RUN/'nominal/trace.npz')==decision['nominal_trace_sha256']
    assert sha(RUN/'post_lifecycle_hold_5s/trace.npz')==decision['extension_trace_sha256']
    final_reports=[]
    for key,item in decision['independent_reports'].items():
        path=windows(item['path']);assert sha(path)==item['sha256'];r=read(path)
        if key.endswith('physics'):assert r['independent_segment_pass'] and r['requested_segment_completed']
        else:assert r['requested_segment_quiet_pass'] and r['intended_segment_completed']
        if key=='nominal_intent':assert r['full_lifecycle_source_intent_pass'] and r['source_metrics']['source_controls']==819
        final_reports.append(path)
    assert len(final_reports)==4
    target=BASE/'source_snapshot_v1'
    assert not target.exists() and not (BASE/'collector_frozen_inputs.json').exists(),'Existing frozen collector must remain preserved.'
    shutil.copytree(BASE/'source_draft_v1',target)
    previous=read(TASK/'fresh_expert_labels_resume_v1/collector_frozen_inputs_v2.json')
    inputs={name:digest for name,digest in previous['input_sha256'].items()
        if '/sonic23_teleop_resume_20260911/' not in name}
    paths=[args.qualification,Path(__file__),BASE/'run_collection_durable.ps1',*final_reports]
    paths += [RUN/name for name in ('nominal/trace.npz','nominal/report.json','nominal/request.json',
        'post_lifecycle_hold_5s/trace.npz','post_lifecycle_hold_5s/report.json','post_lifecycle_hold_5s/request.json',
        'frozen_inputs.json','prelaunch_clearance.json','inputs/baseline250_trace.npz','inputs/precontrol250.npz',
        'initial_seed/fresh_bfm.npz','initial_seed/report.json','preserved_baseline250_prefix_equality.json')]
    paths += [TASK/name for name in ('original_bfm_entry250_v1/entry250/report.json',
        'original_bfm_entry250_independent_physics_v1/report.json','original_bfm_entry250_independent_intent_v1/report.json',
        'fast_controller_nominal_pilot_v1/labels/labels.npz','fast_controller_nominal_pilot_v1/labels/report.json',
        'fresh_expert_labels_resume_v1/labels/labels.npz','fresh_expert_labels_resume_v1/labels/report.json',
        'fast_controller_aggregate_fit_v1/fit/teacher_fit.npz','fast_controller_continued_fit_v1/fit/teacher_fit.npz')]
    branch=read(RUN/'frozen_inputs.json')
    for name,digest in branch['source_sha256'].items():
        path=RUN/'source_snapshot_v1'/name;assert sha(path)==digest;paths.append(path)
    for path in paths:inputs[path.as_posix()]=sha(path)
    for name,digest in inputs.items():assert sha(windows(name))==digest,name
    result=dict(kind='qualified_BFM250_actual_expert_labels_and_phase_compatibility',
        source_sha256={p.relative_to(target).as_posix():sha(p) for p in sorted(target.rglob('*')) if p.is_file()},
        input_sha256=inputs,expert_branch_receipt_sha256=sha(RUN/'frozen_inputs.json'),
        expert_launch_clearance_sha256=sha(RUN/'prelaunch_clearance.json'),
        baseline250_sha256=sha(RUN/'inputs/baseline250_trace.npz'),
        qualification_receipt_sha256=sha(args.qualification),samples=1019,eligible_controls=[250,1268],
        prefix_labels=0,prefix_actor_inference_calls=0,actor_inference_calls_when_executed=1019,
        model_fitting_authorized=False,physics_steps=0,hardware_authorized=False)
    (BASE/'collector_frozen_inputs.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(receipt_sha256=sha(BASE/'collector_frozen_inputs.json'),sources=len(result['source_sha256']),inputs=len(inputs))))

if __name__=='__main__':main()
