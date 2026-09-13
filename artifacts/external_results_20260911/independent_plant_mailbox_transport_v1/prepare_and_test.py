"""Freeze source and run only synthetic spawned-process transport tests."""
import hashlib,json,platform,shutil,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def main():
    snapshot=BASE/'source_snapshot_v1';snapshot.mkdir(exist_ok=False)
    names=['clock_core.py','history.py','mailbox.py','shared_mailbox.py','test_transport.py']
    for name in names:shutil.copy2(BASE/'source_draft_v1'/name,snapshot/name)
    sources={name:sha(snapshot/name) for name in names}
    original=ROOT/'independent_plant_clock_foundation_v1/source_draft_v3'
    unchanged={name:sources[name]==sha(original/name) for name in names[:3]}
    if not all(unchanged.values()):raise ValueError('Original foundation changed')
    pins={str(original/name):sha(original/name) for name in names[:3]}
    for p in [ROOT/'independent_native_stepper_v1/source_draft_v3/native_stepper.py',ROOT/'independent_native_stepper_root_review_v2/review_v3.json',ROOT/'independent_native_stepper_equivalence_root_review_v1/replay_review.json',ROOT/'independent_native_stepper_equivalence_v1/replay_completion_verification.json',BASE/'PROTOCOL.md',Path(__file__)]:pins[p.as_posix()]=sha(p)
    write(BASE/'test_request.json',dict(kind='fixed_synthetic_spawn_transport_tests',source_sha256=sources,input_sha256=pins,
        command=[sys.executable,'-m','unittest','-v','test_transport'],start_method='spawn',task_model_calls=0,native_steps=0,optimizer_updates=0,plant_ticks=0,real_clock_runs=0))
    started=time.perf_counter();result=subprocess.run([sys.executable,'-m','unittest','-v','test_transport'],cwd=snapshot,capture_output=True,text=True,timeout=180)
    (BASE/'stdout.log').write_text(result.stdout,encoding='utf-8');(BASE/'stderr.log').write_text(result.stderr,encoding='utf-8')
    exact=all(sha(path)==digest for path,digest in pins.items()) and all(sha(snapshot/name)==digest for name,digest in sources.items())
    report=dict(source_preparation_passed=result.returncode==0 and exact,synthetic_tests_passed=result.returncode==0,exit_code=result.returncode,elapsed_seconds=time.perf_counter()-started,
        python=sys.version,executable=sys.executable,platform=platform.platform(),request_sha256=sha(BASE/'test_request.json'),source_sha256=sources,input_sha256=pins,
        original_foundation_byte_exact=unchanged,all_final_hashes_exact=exact,stdout_sha256=sha(BASE/'stdout.log'),stderr_sha256=sha(BASE/'stderr.log'),
        task_model_calls=0,native_steps=0,optimizer_updates=0,plant_ticks=0,real_clock_runs=0,real_time_qualified=False,actual_integration_selected=False)
    write(BASE/'source_test_receipt.json',report);print(json.dumps(report));print(result.stderr)
    if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()
