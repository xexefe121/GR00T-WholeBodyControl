# Saved PICO internal-source-state experiment — REJECTED, partial tracking gains

No deployment readiness. No robot command, physical transport, training update,
controller promotion, native limit/gain change, source edit or hidden retry.
All new work is inside the separate sonic-transfer-23dof repository.

## What changed

An affine missing-axis dynamics predictor fitted to existing source29 PICO
data failed all three fit-excluded walking checks. Its position fit improved,
but velocity and leg-command agreement worsened. `calibration_v1/report.json`
records the rejection. It never controlled native23. A first loader KeyError
before fitting/output is preserved under `setup_failure_v1/`.

A distinct physics-based predictor uses the pinned source29 MuJoCo model and
captured source C++ PD parameters. At each control it receives only CURRENT
measured native root/retained23 q/dq plus the full frozen source proposal. Its
six absent q/dq states persist internally, never borrowed from true missing
states or written into the actual23 plant. All ten history frames contain
unchanged real retained feedback and explicitly hypothetical missing states.
The original normal encoder/FSQ/decoder,920ms received-source buffer, unchanged
native source-action codec, range preview, PD gains and motor limits remain.

Source-only torque caps differ from native caps and between older walking/PICO
source experiments. Each identity is pinned; no source cap became a native rating.
Prediction is privileged offline SIM research, not a deployable state estimator.

## Verified results

Source-only preflight31855 EXIT0: all6530 PICO,1417 walk002,1569 walk003 and1114
walk008 controls reproduce missing source joint position AND velocity exactly.
Initial virtual states are zero; no missing truth input or mid-trial reset.
Existing source29 tracking and ankle-range failures remain documented. Accurate
prediction of their dynamics does not make those controllers accepted teachers.

Focused5187 EXIT0:26 tests passed15.08s. Mapping, chronological history, measured
state isolation, invalid input and terminal preview/prediction/range failure
tested. Scoped Ruff E/F passes for the new implementation/tests/drivers.

Native trial49346 writes complete failed evidence and exits0. This is successful
EVIDENCE generation, not successful tracking. Physical controls1629/6530; source
controls1279/5780 =25.58/115.60s. Attempt index1629 (1-based1630) is not integrated.
The separate virtual predictor reaches waist_roll_joint0.525826856rad against
its0.52rad upper bound; the unchanged new-candidate guard rejects it.

Actual23 physical measured hard-range excess0, commanded/engine effort excess0,
maximum velocity/rating ratio0.347930385. All16,290 actual physical substeps
reintegrate from saved applied targets with qpos/qvel differences exactly0.
`native_actual_v1/physics_audit.json` records this independent check.

Same25.58s source prefix (NOT a full-source score), candidate versus baseline:

- Root world position p95:0.228086 versus0.813480m.
- Leg joint RMSE:0.124231 versus0.124182rad; no improvement.
- Arm joint RMSE:0.261937 versus0.295870rad; approximately11.5% lower.
- Pelvis-centered ankle-origin p95 left/right:0.097457/0.123552 versus
  0.117022/0.131525m. Foot contacts/slip are not qualified by ankle origins.
- World ankle-origin p95 left/right:0.225429/0.231147m, still above unchanged
  0.05m screens. Some original relative hand-point/orientation metrics worsen.

Initial standing and acquisition-ramp durations complete, but motion/return
does not. Initial reference-pose reset is not proof of normal-standing acquisition.
This controller variant has no completed pause/recovery or real-time trial.
Older separate runtime fault tests do not automatically qualify it.

Audit53644 EXIT0 independently reconstructs all1630 history vectors and causal
internal forecasts.96 freshly loaded original-source decoder/encoder checks
match raw29/token64 bit-exact. No native policy rerollout occurs in this audit.
`native_actual_v1/history_and_tracking_audit.json` contains per-joint failure
state, identical-prefix foot/arm/leg metrics and source hashes.

## Failure interpretation

The INTERNAL waist oscillates up to11.4265rad/s; just before failure its position
is0.481608rad and velocity+4.84696rad/s. Frozen policy requests target-3.89365rad;
the internal source actuator already brakes at its-50Nm cap. This is not a
measured physical23 waist-roll excursion; that joint does not exist on native23.

Simply clipping that internal target to-0.52rad is not an identified cure: at
the saved starting state, source PD request changes from-133.495Nm to-37.342Nm,
reducing rather than increasing braking after the50Nm cap. This arithmetic
check is not a new dynamics trial or a global proof against constrained control.
No clipping sweep, virtual reset, wider range or increased torque was run.

Root/arm prefix improvements justify preserving the measured mechanism, but
do not fix missing-state stability, leg fidelity or deployment. A further
candidate needs evidence addressing those failures, not a longer unguarded run.
Goal remains active. Physical G1 damping and mode handback remain untested here.

## Measured video

Render2235 EXIT0: `native_actual_v1/rejected_virtual_model.measured.mp4` contains
all1630 measured state boundaries,50Hz,H264,960x720,32.60s including initial
frame. Actual integration duration32.58s includes7s standing/ramp and25.58s
source. No extra policy or physical dynamics run for rendering; this shows the
failed measured native23 trajectory, not the hypothetical source29 plant.

Camera follows root, so video is NOT a world-tracking comparison. Frames750
(15.00s) and1629 (32.58s) were extracted and visually inspected. Both show native
robot limbs clearly; no claim that those images demonstrate tracking acceptance.
First frame extraction command failed because PowerShell split `-frames:v`;
retry with `-vframes` succeeded without overwriting any artifact. No controller
or dynamics retry occurred. `render.json` binds video, report and trace hashes.

Video SHA256:
`73012f9cbc807ee3469d313bd4c01c4db249b4ee81e774dc1e978acf2f07ecf7`.
All experiment, test, audit and render jobs for this continuation are terminal.

## Recorded commands and primary evidence

WSL Ubuntu-22.04, repository root; existing local assets only:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=.:/mnt/z/codex/GR00T-WholeBodyControl/external_dependencies/unitree_rl_mjlab
/root/.venvs/g1_true23_mjlab/bin/python artifacts/g1_true23_virtual_state_20260910_v1/calibrate.py
/root/.venvs/g1_true23_mjlab/bin/python artifacts/g1_true23_virtual_state_20260910_v1/verify_physics_predictor.py
/root/.venvs/g1_true23_mjlab/bin/python -m pytest -q gear_sonic/tests/test_g1_true23_virtual_source_history.py gear_sonic/tests/test_g1_true23_discarded_action_memory.py gear_sonic/tests/test_released_core_comparison.py
/root/.venvs/g1_true23_mjlab/bin/python artifacts/g1_true23_virtual_state_20260910_v1/run_native.py
/root/.venvs/g1_true23_mjlab/bin/python artifacts/g1_true23_virtual_state_20260910_v1/audit_native.py
```

Completed evidence directories refuse overwrite. These are recorded commands,
not instructions to silently delete/retry a rejected trial. Reproduction needs
a separately named evidence copy retaining these exact sources and all pins.

Native report SHA256:
`363d7d9f9b091cbe0f60fec5aed69f14934804230db6b4245f3c5373b9a5649f`.
Source-only preflight report SHA256:
`f46589e6d28184b26eff4b120a625b2e5a032bf084c348f0f44f05501195f02b`.
The report/started receipts bind executed implementation sources and inputs.
