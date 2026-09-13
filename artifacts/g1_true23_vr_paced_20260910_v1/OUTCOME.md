# Actual-clock recorded-input SIM: fault handling works; uninterrupted timing fails

Goal remains ACTIVE. This is runtime progress, not a leg-tracking fix or a
deployment-ready policy. Original implementation, physical robot, policy
weights, gains, effort/range gates and source motion remain unchanged.

## Executed outcomes

- `end-of-stream-v1`: startup rejected122.045ms-old input before any physics.
  Model hashing was performed after subscription. `startup_profile.json`
  measured231–291ms hashing versus0.29–0.59ms packet validation, with zero physics.
  Preparation now finishes before subscribing. Failed source versions and
  request/failure evidence remain preserved in that case directory.
- `end-of-stream-v2`: full requested906 controls integrated, but only394 SONIC
  plus512 balance. At zero-based control393, execution25.336ms exceeded the20ms
  budget. The scheduler failure latched balance and rebased its clock without
  catching up in bursts. This is a FAILED full-source real-time attempt. It
  never reached the planned EOF transition while still tracking the source.
- `pause-v1`:500ms source pause starting at control200.200 SONIC +706 balance,
  first fault exactly200, all18.12s wall time completed. No timing misses.
- `gap-v1`: source packet200 omitted. Same200+706 result; gap detected at200.
- `payload-v1`: malformed JSON replaces packet200. Same200+706 result; payload
  fault detected at200. Neither case silently restarts when later input arrives.

Maximum control execution for pause/gap/payload:18.640/15.877/17.588ms. Maximum
admitted source ages41.029/41.194/41.600ms; unchanged freshness limit100ms.
Each fault run ends with maximum measured joint speed0.000520rad/s. The timing-
failure run ends at0.002026rad/s. This is the compatibility balance actor, not
physical Unitree normal-standing firmware or ownership handback.

All published message hashes arrived in order:656/631/655/656 messages for
the four executed physics cases respectively. Message omissions in pause/gap
are explicit publisher faults, not hidden crops. Each case retains sender and
receiver timestamps. `sent_ns` is sampled after send() returns, so it must not
be treated as an exact timestamp before the receiver could observe the message.

## Independent verification

Focused regression81432 EXIT0:45 tests pass12.87s across new paced scheduling,
virtual clocked controls and the old paired-live/transport contracts. Scoped
Ruff E/F passes. Tests cover actual-versus-virtual packet age, bad input, FIFO
overflow, early/late wake, compute/capture overruns, no catch-up or implicit
restart, and expensive preparation before packet receipt. All experiment,
profiling and verification jobs are terminal.

`audit_paced.py` / `verification.json` verify104 source pins per case, report/
trace/message hashes,3,624 controls and36,240 actual2ms substeps. Recomputed
post-substep positions/velocities match recorded control states exactly. All
virtual physics timestamps, real scheduling arithmetic, missed deadlines,
packet-age bounds and latched balance masks are checked from saved arrays.

All four cases: measured joint-range excess0, commanded effort ratio<=1,
maximum motor velocity ratio0.569275. No torque-slew, contact/slip, disturbance
or hardware qualification is implied. This audit does not reintegrate new
dynamics or retroactively make the timing-failure case pass.

Every pre-fallback SONIC prefix is bit-exact to the prior walking baseline.
The new full virtual capture in `profile_inference.py` also matches all656
controls of that old baseline exactly. Known leg RMSE0.207371rad, foot-position
p950.266102/0.208334m and arm RMSE0.596487rad therefore remain failures. The
actual streamed full walking motion has not passed uninterrupted execution.

## Inference and prior hypothesis checks

`profile_inference.py` collects one complete unchanged virtual walking trace
and656 actual encoder/history input pairs. It then runs1,968 same-input forward
comparisons. Default, no-spinning, and single-thread/no-spinning CPU sessions
have bit-exact raw outputs and token/history arrays. Inference p95 timings are
5.600/6.546/8.514ms; maximum7.263/8.426/14.845ms in that microbenchmark. Alternatives
were slower and were not adopted. Total controller work and OS scheduling are
not measured by inference alone. Python GC captured during the virtual trace
has maximum duration0.0333ms; this is not evidence for the prior25ms outlier's
cause. No repeated full-stream attempt was run merely to obtain a green result.

The initial learned-codec hypothesis was incorrect: original transfer uses an
[analytic codec with decoder adapters](https://sonic-agibot-x2.github.io/sonic-transfer/).
Existing local lower240 perturbation results already show changing leg tokens.
Fresh read-only normal64 critic/return calculation gives explained variance
0.91704 at update100, so its raw value loss does not establish a broken critic.
No new policy fit or architecture change was justified by those checks.

## Reproduce a separately saved actual-stream case

Use the existing WSL environment. The harness binds only a random localhost
port, waits for subscription acknowledgement, then publishes the existing full
clip at50Hz. Do not reuse an output directory; overwrite is refused.

```bash
cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:/mnt/z/codex/GR00T-WholeBodyControl/external_dependencies/unitree_rl_mjlab \
  /root/.venvs/g1_true23_mjlab/bin/python -m gear_sonic.scripts.test_g1_true23_paced_saved_stream \
  --repository-root /mnt/z/codex/GR00T-WholeBodyControl \
  --encoder-report artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.encoder.json \
  --decoder-report artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.decoder.json \
  --packets /mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/walk002/causal_packets.json \
  --scenario end-of-stream \
  --output-directory artifacts/g1_true23_vr_paced_manual_01
```

Other scenarios: `pause`, `gap`, `payload`. Exit0 denotes the declared scenario,
timing and limited physical screens only; tracking/deployment remain false.
The generic consumer `run_g1_true23_paced_sim` additionally accepts an existing
localhost endpoint and bounded source-control count. Its exit code denotes
limited physical/timing checks, not full-source or hardware qualification;
inspect `full_source_sonic_completed` and input/runtime faults in the report.

The consumer requires source timestamps in its own host monotonic clock domain.
Only startup input waiting can block before the simulator is initialized. Once
running, a missing packet cannot postpone the next physics tick. Actual-age
validation supplements exact virtual timestamps. A40ms initial buffer is
explicit latency, not a claim of zero-latency teleop. Ten2ms substeps remain
batched inside each wall-clock20ms control; no hard-real-time OS claim is made.

## Remaining work

This does not solve source tracking, ordinary-standing acquisition, deliberate
SONIC re-entry, headset sensing/root estimation or physical handback. A recorded
reference from TWIST2's PICO workflow is not a raw live headset session. No robot
commands, policy promotion, gain/limit changes, commits, pushes or new source
downloads occurred. No new video was rendered or old video relabelled.

Next timing diagnosis needs stage-level measurement of the whole control path;
changing inference threads or raising the20ms budget is not a measured fix.
Tracking needs a separate evidence-backed controller/reference intervention;
do not repeat the rejected100-update recipes, absent-joint deletion, input-order
or gain sweeps as new ideas. All prior fidelity limits stay visible.
