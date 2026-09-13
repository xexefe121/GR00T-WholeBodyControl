"""Saved timestamps, source and manifest sizes only; no timing replay."""
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'direct_target_student_evaluation_v1'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def time(s):return datetime.fromisoformat(re.sub(r'(\.\d{6})\d+',r'\1',s).replace('Z','+00:00')).timestamp()
def mtime(p):return Path(p).stat().st_mtime
def category(path):
    p=str(path).replace('\\','/').lower()
    if '/velocity_chord_student_v1/generation/' in p and not p.endswith('/centers.npz'):
        return 'known_transitive_velocity_generation'
    if '/one_step_policy_branch_collection_resume2969_v1/collection/data/' in p:
        return 'known_transitive_physical_branch_data'
    if '/pico_walk002_labels_v1/collection/' in p:
        return 'known_transitive_broader_training_labels'
    if '/one_step_physical_student_v1/fit/' in p:
        return 'known_transitive_prior_fit_sampling_and_evidence'
    return 'other_pins_kept_conservative'

def stage(mode):
    f=BASE/(mode+'_process');launch=read(f/'launch_receipt.json');binding=read(BASE/(mode+'_binding.json'))
    owner=read(BASE/(mode+'_completion_verification.json'))
    assert owner['owner_completion_accounting_passed'] is True
    assert owner['pins_exact']==len(launch['input_hashes'])
    assert read(f/'postrun_hashes.json')==launch['input_hashes']
    dispatch=read(f/'dispatch.json');start=read(f/'start.json');raw=read(f/'raw_exit.json');end=read(f/'exit.json')
    timings=dict(dispatch=time(dispatch['utc']),wrapper_after_initial_pin_check=time(start['utc']),
                 child_started=mtime(f/'child.json'),raw_python_exit=time(raw['utc']),wrapper_exit=time(end['utc']))
    if mode=='evaluation':
        timings.update(initial_snapshot=mtime(BASE/'canonical_initial_snapshot.npz'),
            main_report=mtime(BASE/'nominal/report.json'),skipped_hold_report=mtime(BASE/'post_lifecycle_hold_5s/report.json'),
            final_pilot_outcome=mtime(BASE/'pilot_outcome.json'))
        main=read(BASE/'nominal/report.json')
        reported_elapsed=main['elapsed_seconds']
        intervals=dict(dispatch_to_first_pin_check_finished=timings['wrapper_after_initial_pin_check']-timings['dispatch'],
            child_start_to_initial_snapshot=timings['initial_snapshot']-timings['child_started'],
            initial_snapshot_to_main_report=timings['main_report']-timings['initial_snapshot'],
            main_report_to_pilot_outcome=timings['final_pilot_outcome']-timings['main_report'],
            raw_exit_to_wrapper_exit=timings['wrapper_exit']-timings['raw_python_exit'])
    else:
        timings.update(last_call_counter_write=mtime(BASE/'head_witness/attempt.json'),
            witness_report=mtime(BASE/'head_witness/report.json'))
        reported_elapsed=None
        intervals=dict(dispatch_to_first_pin_check_finished=timings['wrapper_after_initial_pin_check']-timings['dispatch'],
            child_start_to_last_call_counter_write=timings['last_call_counter_write']-timings['child_started'],
            last_call_counter_write_to_witness_report=timings['witness_report']-timings['last_call_counter_write'],
            raw_exit_to_wrapper_exit=timings['wrapper_exit']-timings['raw_python_exit'])
    files=[];groups={}
    for path,digest in launch['input_hashes'].items():
        size=Path(path).stat().st_size;kind=category(path)
        files.append(dict(path=path,expected_sha256=digest,bytes=size,category=kind))
        g=groups.setdefault(kind,dict(files=0,bytes=0));g['files']+=1;g['bytes']+=size
    launch_bytes=sum(v['bytes'] for v in groups.values())
    binding_bytes=sum(Path(e['path']).stat().st_size for e in binding['input_files'])
    known=sum(v['bytes'] for k,v in groups.items() if k.startswith('known_transitive_'))
    elapsed=timings['wrapper_exit']-timings['dispatch']
    result=dict(mode=mode,source='saved UTC records plus filesystem write times; no profiler',
        raw_python_exit=raw['raw_python_exit_code'],diagnostic_wrapper_exit=end['exit_code'],
        wall_seconds=elapsed,reported_segment_elapsed_seconds=reported_elapsed,
        unassigned_to_reported_segment_seconds=None if reported_elapsed is None else elapsed-reported_elapsed,
        timestamps_utc={k:datetime.fromtimestamp(v,timezone.utc).isoformat() for k,v in timings.items()},
        observed_intervals_seconds=intervals,hash_pass_groups=groups,launch_pin_files=len(files),launch_pin_bytes=launch_bytes,
        known_transitive_training_bytes=known,known_transitive_fraction=known/launch_bytes,
        binding_pin_files=len(binding['input_files']),binding_pin_bytes=binding_bytes,
        guaranteed_five_bulk_hash_pass_bytes=2*launch_bytes+3*binding_bytes,
        five_pass_derivation=['durable prelaunch receipt pins','child binding pins','driver initial require_ready input_files',
                              'driver final require_ready input_files','durable postrun receipt pins'],
        excluded_hash_work='extra named/source/review reads; freezing, launch generation, review, owner verification',
        files=files)
    return result

def main():
    assert not (HERE/'report.json').exists()
    witness=stage('witness');evaluation=stage('evaluation')
    inputs=[BASE/'freeze_final_package.py',BASE/'prepare_bound_launcher.py',BASE/'source_draft_v1/evaluation_gate.py',
            BASE/'source_draft_v1/evaluate_direct_target_student.py',BASE/'source_draft_v1/head_activation_witness.py',
            BASE/'nominal/report.json',BASE/'post_lifecycle_hold_5s/report.json',BASE/'pilot_outcome.json',Path(__file__)]
    for mode in ('witness','evaluation'):
        inputs.extend([BASE/(mode+'_binding.json'),BASE/(mode+'_completion_verification.json')])
        inputs.extend(BASE/(mode+'_process')/name for name in ('dispatch.json','start.json','child.json','raw_exit.json',
            'exit.json','launch_receipt.json','postrun_hashes.json','run.ps1','run_durable.ps1'))
    pins={p.resolve().as_posix():sha(p) for p in inputs}
    report=dict(kind='direct_target_launch_saved_timing_assessment',passed=True,witness=witness,evaluation=evaluation,
        input_sha256=pins,model_calls=0,native_steps=0,optimizer_updates=0,downloads=0,
        conclusions=[
            'Five complete bulk pin-hash passes occur in each launch, beyond preparation/review/owner checks.',
            'Measured intervals establish wall overhead; the package has no per-gate or per-file profiler, so exact hash CPU/I/O time cannot be separated from process/WSL/model initialization and artifact writes.',
            'Known transitive generation/physical/broader-label files are hashed by readiness but never consumed by direct feature or inference arithmetic.',
            'This failed partial rollout does not measure full-lifecycle or live deadline performance.'
        ],
        proposed_future_package=[
            'Use one immutable completed fit qualification receipt with direct training-manifest, final export, normalization/output-contract, and data-review subject hashes; keep full training verification in the training/data stage.',
            'Pin the actual ONNX, WSL runtime binaries, native model and dependent meshes/contract/options, imported controller/helpers, selected clip/timeline/reference/source goals, initial full291 fixture and required parity inputs.',
            'Prepare a separately reviewed minimal control250 parity capsule containing the exact required saved rows and native endpoint/history/prior, bound to original qualified artifacts; do not silently change current whole-archive gates.',
            'Keep actual runtime inputs under pre/post verification, preserve one-shot locks and raw/diagnostic exits, and timestamp each validation/initialization/segment stage.',
            'Do not modify or relaunch the completed current evaluation; new package requires explicit source review.'
        ],
        limitations=['Size classification is deliberately conservative; other_pins may include more unused training evidence.',
            'File sizes were stat-read after owner verified every content hash; corpus contents were not rehashed solely for this diagnosis.',
            'Filesystem write timestamps are wall-clock observations, not monotonic profiler measurements.'])
    for p,d in pins.items():assert sha(p)==d
    (HERE/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(report_sha256=sha(HERE/'report.json'),wall_seconds=evaluation['wall_seconds'],
        reported_segment_seconds=evaluation['reported_segment_elapsed_seconds'],
        known_transitive_GiB=evaluation['known_transitive_training_bytes']/2**30,
        total_pin_GiB=evaluation['launch_pin_bytes']/2**30,
        guaranteed_hash_read_GiB=evaluation['guaranteed_five_bulk_hash_pass_bytes']/2**30)))

if __name__=='__main__':main()
