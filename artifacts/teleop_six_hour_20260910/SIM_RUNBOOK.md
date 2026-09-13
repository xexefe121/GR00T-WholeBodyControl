# Native23 simulation evidence

This workspace contains running native 23-actuator physics and candidate
controllers. Full-body teleoperation qualification is still under test. See
`SESSION.md` for the latest measured outcome; a completed program or a short
successful segment does not mean all qualification gates passed.

The physical model has 12 leg joints, one waist joint and five joints per arm.
MuJoCo 3.2.3 advances at 500 Hz; targets update at 50 Hz. Simulation runs use
native joint, speed and effort limits. No robot, DDS or Pico interface is used.

## Run the tested walking and quiet-standing demo

In PowerShell:

```powershell
& 'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof\artifacts\teleop_six_hour_20260910\RUN_PASSING_WALK.ps1'
```

This launcher was exercised end to end. It creates a fresh simulated plant,
replays every walking and return target, switches to native BFM for terminal
standing, and tests a separate 5-second hold. It checks all15 recorded-source
gates plus both quiet-standing windows and exits with an error if they fail.
Its pinned input hashes prevent changed helper code or data from silently
replacing the verified case. Imports use isolated frozen helper files under
`E:/codex-artifacts/sonic23_teleop_resume_20260911/preserved_walk_demo_v1/repo`
while controller development continues. Every invocation uses a fresh output
directory; existing evidence is never overwritten. The launcher mounts WSL E:
when needed in the same invocation; it requires the existing Z: mount and
pinned runtimes described below.

The checked launcher output is
`E:/codex-artifacts/sonic23_teleop_six_hour_20260910/bfm_online_intent_v2/walk003_quiet_frozen_import_verification_v3`.
The full36.38-second video is
`E:/codex-artifacts/sonic23_teleop_six_hour_20260910/visual_walk003_terminal_bfm_yaw4_full_v1/full_lifecycle_and_separate_hold.fixed_world.mp4`.
This is an offline saved-plan walking demonstration with a tested terminal
controller; it does not run live MPC or use received Pico tracking.

## Reproduce the complete nominal walking pass

`walk003` completed all 819 source controls and the entire 31.38-second
lifecycle. An independent manual-PD replay in the pinned WSL MuJoCo 3.2.3
runtime reproduced every physical state and torque exactly and passed all 15
recorded-source gates. Standing still has residual joint motion, and the
producer took seconds per planning block. This establishes one nominal
offline result, not live teleoperation or a settled standing controller.

The complete video is:
`E:/codex-artifacts/sonic23_teleop_six_hour_20260910/visual_walk003_allmargin_nominal_full_v1/full_source_and_lifecycle.fixed_world.mp4`.

Run the following PowerShell commands with the existing `/mnt/e` and `/mnt/z`
WSL mounts. The frozen runner is specific to this full walk003 case. It
creates a new simulated plant and applies actual saved targets with native
PD; no state copying occurs after initialization. The reproduction output
must not already exist. These commands use the same runner and inputs as the
completed independent replay; the new output name has not been run again.

```powershell
$archiveWsl = '/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910'
$reproductionWsl = "$archiveWsl/bfm_online_intent_v2/walk003_allmargin_wsl_reproduction_001"
wsl.exe -e env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 `
  /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python `
  "$archiveWsl/bfm_online_intent_v2/walk003_allmargin_wsl_independent_replay_v1/runner_snapshot.py" `
  --bundle /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1 `
  --reference "$archiveWsl/mjbatch_intent_floor_inputs_v1/walk003/reference.npz" `
  --producer "$archiveWsl/mjbatch_full_v1/walk003_v4_native323_allmargin_full_v1" `
  --output $reproductionWsl
if ($LASTEXITCODE -ne 0) { throw 'Independent replay failed' }

Set-Location -LiteralPath 'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof'
$archiveWin = 'E:\codex-artifacts\sonic23_teleop_six_hour_20260910'
$reproductionWin = "$archiveWin/bfm_online_intent_v2/walk003_allmargin_wsl_reproduction_001"
& 'C:\Users\camer\AppData\Local\Programs\Python\Python310\python.exe' `
  -m artifacts.teleop_six_hour_20260910.qualify_recorded_candidate `
  $reproductionWin `
  --motion-override "$archiveWin/mjbatch_intent_floor_inputs_v1/walk003/reference.npz" `
  --output "$reproductionWin/recorded_source_audit_v2.json" --require-pass
if ($LASTEXITCODE -ne 0) { throw 'Recorded-source acceptance failed' }
```

The existing independently checked result is under
`bfm_online_intent_v2/walk003_allmargin_wsl_independent_replay_v1`.
Fixed-feedback replay on Windows diverged and failed; use of the same MuJoCo
version alone does not establish feedback robustness across environments.

## Reproduce an independently rejected physical replay

Run this in PowerShell on the current machine. It starts a new simulated plant
from the original initial state, then applies saved targets with measured-state
feedback and native PD. Recorded states are feedback references, never copied
into the physical plant after initialization.

```powershell
$repoPath = 'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof'
$evidencePath = 'E:\codex-artifacts\sonic23_teleop_six_hour_20260910'
$pythonPath = 'C:\Users\camer\AppData\Local\Programs\Python\Python310\python.exe'
$runOutput = Join-Path $evidencePath ('independent_walk002_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
Set-Location -LiteralPath $repoPath
& $pythonPath -m gear_sonic.scripts.evaluate_g1_true23_mjbatch_plan_replay `
  --plan "$evidencePath/mjbatch_full_v1/walk002_native323_h30_clip1_v1" `
  --mode feedback --feedback-clip 0.1 --output $runOutput
& $pythonPath -m artifacts.teleop_six_hour_20260910.qualify_recorded_candidate `
  $runOutput --output "$runOutput/recorded_source_audit.json" --require-pass
```

This particular complete walking trial is **expected to fail** the strict gate:
left relative-foot p95 is 0.13685 m against a 0.12 m limit, and actual ankle
position exceeds its native bound by 0.001825 rad. The auditor exits with
status 1 for a failed or incomplete candidate. Its JSON names each failed gate.
The replay itself is fast because its expensive planning was already done.
It is not an online MPC timing result.

For retargeted plans, additionally pass their exact local reference file with
`--motion-override`. The referee checks the original source, model, physics,
reference and receipt hashes. Existing output directories are never overwritten.

## Received-input and timing component

`gear_sonic.scripts.evaluate_g1_true23_bfmzero_stream` tests an actual
received-only buffer, packet faults, standing transitions and paced simulation.
The existing quiet-machine evidence is under
`bfm_online_intent_v2/stream_quiet_normal_v1`: 1,824 complete control cycles,
zero missed 20 ms deadlines, measured loop p95 9.616 ms and maximum 16.451 ms.
The entire raw sensor/history/action/physics trace was independently checked.

That fast controller still fails full-body source tracking. The MPC controller
has better tracking on some cases, but takes seconds per 100 ms planning block.
These are separate components; their successes cannot be combined into an
untested full-body real-time claim.

## Offline planner and provenance

The user's [mjbatch repository](https://github.com/kevinzakka/mjbatch) is pinned
at `77966f85bcd8f7ef4351cb4a1a6f42e133d19725`. Its batched derivatives and iLQR
math were adapted to the native23 model. The upstream 29-DOF example robot and
cropped/lifted reference are not the robot used here. Runtime/build checks are
in `MJBATCH323_ENVIRONMENT.md`; native PD/affine-actuator parity is independently
verified. The isolated WSL interpreter is:

```
/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python
```

`gear_sonic.scripts.evaluate_g1_true23_mjbatch_mpc` writes request/model/source
hashes, frozen source files, every 2 ms physical state/effort, optimization logs
and atomic incomplete checkpoints. Each experiment records its own settings;
different cost or seed variants are not silently treated as one controller.

For H30 with the recorded BFM seed, preview is 740 ms of reference packets and
up to 760 ms of raw poses because central velocities need another source pose.
The immutable request's historical 740 ms field is clarified by
`preview_support_supplement_v2.json`; original request hashes are unchanged.

## Acceptance and inspection

`SIM_ACCEPTANCE.md` defines full original-source/lifecycle completion, root,
heading, both feet, all 12 leg joints, original hand/head intent and every-2ms
physical limits. Original root and heading remain authoritative after reference
retargeting. Reference floor clearance alone does not establish feasible contact
or dynamic balance.

`qualify_recorded_candidate.py` checks those recorded-source gates and always
keeps online, sensor-only, hardware and full teleoperation qualification false.
`--require-pass` makes failed/incomplete recorded-source evidence a nonzero exit.
Engine warning evidence must be present and clear; malformed torque arrays are
rejected. Negative verification fixtures are explicitly labelled as altered
fixtures and are not controller trials.

`inspect_mpc_checkpoint.py` scores an immutable prefix without claiming a full
result. `materialize_mpc_checkpoint.py` packages such a prefix for an independent
physical replay, preserving its original bytes and incomplete status. It does
not resume planning or fill in missing motion.

Full-body optical recordings are useful simulation input. They do not prove
that live Pico trackers can reconstruct all leg motion. That remains a separate
stage after a controller passes complete simulation tests.
