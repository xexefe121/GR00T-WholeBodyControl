"""Compile and bind one fixed 364-case native bulk diagnostic; no dynamics."""
import hashlib
import json
from pathlib import Path
import subprocess
import mujoco

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    package = Path(mujoco.__file__).resolve().parent
    library = package / 'libmujoco.so.3.2.3'
    assert mujoco.__version__ == '3.2.3'
    assert sha(library) == '78f5455cbcbc2b4c1452f09a096aaa1d79893d63bbeb0bad93f610ddde19d264'
    assert not (HERE / 'native_rollout.so').exists() and not (HERE / 'request.json').exists()
    command = ['/usr/bin/g++', '-std=c++17', '-O2', '-shared', '-fPIC', '-fno-fast-math',
               '-ffp-contract=off', '-fno-tree-vectorize', '-Wall', '-Wextra', '-Werror',
               '-I' + str(package / 'include'), str(HERE / 'native_rollout.cpp'),
               str(library), '-Wl,-rpath,' + str(package), '-o', str(HERE / 'native_rollout.so')]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    (HERE / 'build_stdout.log').write_text(result.stdout)
    (HERE / 'build_stderr.log').write_text(result.stderr)
    inputs = {str(p): sha(p) for p in [HERE / 'native_rollout.cpp', Path(__file__), library, Path('/usr/bin/g++')]}
    inputs.update({str(p): sha(p) for p in sorted((package / 'include/mujoco').glob('*.h'))})
    build = dict(command=command, exit_code=result.returncode, inputs=inputs,
                 native_library=str(library), output_sha256=sha(HERE / 'native_rollout.so') if result.returncode == 0 else None)
    (HERE / 'build.json').write_text(json.dumps(build, indent=2)+'\n')
    if result.returncode:
        raise RuntimeError('compile failed; build logs preserved')
    old = json.loads((BASE / 'phase_student_preallocated_forecast_v2/request.json').read_text())
    inputs = dict(old['input_sha256'])
    additions = [HERE / name for name in ('native_rollout.cpp', 'native_rollout.so', 'bulk_forecast.py',
                                        'build_and_freeze.py', 'run_parity.py', 'build.json')]
    additions += [BASE / 'filtered_all_candidates132_v1/cases.npz',
                  BASE / 'filtered_all_candidates132_v1/request.json',
                  BASE / 'filtered_all_candidates132_v1/results/report.json']
    inputs.update({str(p): sha(p) for p in additions})
    request = dict(kind='fixed364_private_native_bulk_parity', input_sha256=inputs,
                   case_sets=[dict(path=str(BASE / 'phase_student_preallocated_forecast_v2/cases.npz'), count=232),
                              dict(path=str(BASE / 'filtered_all_candidates132_v1/cases.npz'), count=132)],
                   case_count=364, maximum_private_horizon_seconds=.1,
                   reference='independent oracle stop_on_failure=False, full computed horizon',
                   prefix_reference='same full oracle trace first-failure prefix; preserved original232 and132 outcomes',
                   order='fixed alternating bulk/oracle, one run each per case',
                   extra_computed_steps_after_first_failure_retained=True,
                   actual_plant_steps=0, actor_calls=0, optimizer_calls=0, connected_controller=False,
                   do_not_claim_strict_early_stop=True)
    (HERE / 'request.json').write_text(json.dumps(request, indent=2)+'\n')
    print(json.dumps(dict(compiled=True, frozen_cases=364, request_sha256=sha(HERE/'request.json'), build=build['output_sha256'])))


if __name__ == '__main__':
    main()
