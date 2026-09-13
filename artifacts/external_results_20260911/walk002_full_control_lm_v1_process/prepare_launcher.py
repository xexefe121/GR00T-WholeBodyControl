"""Freeze the one authorized canonical walk002 command; no simulation execution."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUTPUT = BASE / 'walk002_full_control_lm_v1'
if OUTPUT.exists() or (HERE / 'pid.json').exists() or (HERE / 'running.json').exists():
    raise SystemExit('existing run/output; never duplicate')


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


assessment_dir = BASE / 'walk002_same_controller_assessment_v1'
assessment = json.loads((assessment_dir / 'assessment.json').read_text())
prior = json.loads((BASE / 'pico_full_control_lm_v1_process/launch_receipt.json').read_text())
proposal = assessment_dir / 'proposed_launch.ps1.txt'
if sha(proposal) != '6bdbfcaaf17510d32fd404bcdf176cc3296c8a5808a8d609f0d8386ca624266b':
    raise ValueError('authorized command changed')
launcher = BASE / 'launch_walk002_full_control_lm_v1.ps1'
launcher.write_bytes(proposal.read_bytes())
wrapper = (BASE / 'pico_full_control_lm_v1_process/run_durable.ps1').read_text()
wrapper = wrapper.replace('pico_full_control_lm_v1', 'walk002_full_control_lm_v1')
wrapper = wrapper.replace('requested_controls=6530', 'requested_controls=1417')
wrapper = wrapper.replace('$report.completed_controls -eq 6530', '$report.completed_controls -eq 1417')
wrapper = wrapper.replace('        $result.completed_controls = $report.completed_controls',
                          '        if (-not $result.completed) { $exitStatus = 1; $result.exit_code = 1 }\n        $result.completed_controls = $report.completed_controls')
(HERE / 'run_durable.ps1').write_text(wrapper)
full_contract = dict(kind='ONE fresh canonical walk002 with qualified PICO controller',
                     requested_controls=1417, source_requested_controls=667, expected_physics_steps=14170,
                     canonical_initialization='new native MjData from unchanged v4 frame10',
                     source_control_interval=[350, 1017], terminal_standing_control_interval=[1117, 1417],
                     original_source_frame_interval=[361, 1028], controls_from_exact_timeline=True,
                     shared_controller_edits=False, source_reference_edits=False, model_edits=False,
                     hard_gates_unchanged=True, first_failure_preserved=True,
                     actual_request_must_match_qualified_config=True,
                     independent_native_and_original_intent_audit_required=True,
                     separate_hold_execution_authorized=False, realtime_qualified=False, hardware_authorized=False)
(HERE / 'full_run_contract.json').write_text(json.dumps(full_contract, indent=2))
paths = [Path(path) for path in prior['verified_hashes']]
paths += [Path(path) for path in assessment['source_asset_hashes']]
paths += [assessment_dir / name for name in ('assessment.json', 'ASSESSMENT.md', 'assess.py', 'proposed_launch.ps1.txt')]
paths += [launcher, HERE / 'run_durable.ps1', HERE / 'full_run_contract.json', Path(__file__)]
hashes = {str(path): sha(path) for path in paths}
for recorded in (prior['verified_hashes'], assessment['source_asset_hashes']):
    for name, expected in recorded.items():
        if hashes[name] != expected:
            raise ValueError('frozen file changed: ' + name)
receipt = dict(utc=datetime.now(timezone.utc).isoformat(),
               scope='ONE fresh walk002 full1417/all667source with unchanged qualified PICO controller',
               exact_launcher_text=launcher.read_text(), verified_hashes=hashes,
               requested_controls=1417, source_requested_controls=667,
               source_receipt_sha256=prior['source_receipt_sha256'],
               assessment_sha256=sha(assessment_dir / 'assessment.json'),
               concrete_launcher_sha256=sha(launcher), durable_wrapper_sha256=sha(HERE / 'run_durable.ps1'),
               duplicate_guard='Output directory/stdout/stderr/PID/running records must be absent before launch',
               process_style='Hidden durable wrapper, .NET SHA256, PID/start/exit receipts',
               no_solver_reference_or_model_changes=True, no_physics_started=True,
               authorized_by_root=True, hardware_authorized=False)
(HERE / 'launch_receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(dict(files=len(hashes), launcher_sha256=sha(launcher),
                      wrapper_sha256=sha(HERE / 'run_durable.ps1'),
                      receipt_sha256=sha(HERE / 'launch_receipt.json'))))
