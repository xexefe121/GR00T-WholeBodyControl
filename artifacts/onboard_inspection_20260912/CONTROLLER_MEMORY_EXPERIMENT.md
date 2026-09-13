# Native23 controller-memory experiment

Current stage: corrected two-seed pilot complete; continuation rejected.
Final isolated replay measurements and full-duration comparison video are complete.
Full-body teleop remains unqualified. Training has stopped; no new recipe was launched.

The six-hour implementation window began 2026-09-13 03:25:45 UTC and ends
09:25:45 UTC. The deadline was not reset when an implementation bug was found.
Results live in
`E:\codex-artifacts\sonic23_teleop_resume_20260911\onboard_factory_firmware_v1\controller_state_ab_v1`.
Only `corrected_pilot` may support the learning decision. The earlier pilot and
its evaluations are marked invalid in `INVALIDATED_INITIAL_PILOT.json`.

## Implemented

- Versioned owned controller snapshots at packet-admitted / pre-observation
  boundary. Received buffers, applied-command history, factory history, gait,
  velocity command, standing transitions and fault/rearm state are restored.
  Duplicate observation timestamps are rejected.
- Expert-prefix reconstruction advances the real observation path and commits
  recorded applied commands. Actor proposals never enter reconstructed memory.
- Native integration restoration keeps float64 state. Available solver states
  are bound to matching source trajectories; missing states use the same fresh
  solver initialization in both arms and do not support exact-continuation claims.
- Shared native preview and standing capture. Preview uses eight workers with
  one native thread per worker. Actual command delay varies from0 to12 ms.
- Complete runtime wrapper/configuration checks. The retained actor bytes are
  unchanged; a separate package pins model, gains/limits, factory configuration,
  preview settings and standing/task reference settings.
- Simulated joint/IMU estimator adapter, including declared noise/bias/delay and
  latched failures. Root position and velocity are not estimator inputs.
- Optional current-pose extension to the existing Pico ZMQ producer, preserving
  original29 hand/head intent. Adapter maps joints by name, retains timestamps,
  and computes derivatives from received history. Legacy producer output remains
  the default.
- Current simulation launcher has recorded/live and ground-truth/estimated
  selections, explicit input rearm and captured-packet replay.

The real SOMA packet uses wrist-roll body names where native23 uses rubber-hand
body names. An explicit mapping preserves their identical joint-frame origins;
the original wrist-yaw hand task points remain unchanged. Local ZMQ transport
and replay pass with the actual Newton-generated packet. Subscription begins
after controller warmup, avoiding queued stale packets at initial arm. Packet
payloads and original timestamps survive recording/replay unchanged; replay
clock rebasing is explicit. This checks transport, not a live headset session.

## Repairs required before a valid comparison

`mjbatch.forward()` refreshes all worlds. The first implementation restored
solver memory only for reset worlds, inadvertently changing unreset worlds.
The scoring refresh also advanced solver memory. Both now preserve
`qacc_warmstart` across derived-state refreshes. Both seeds pass a reset-order
check with exact integration states and exact subsequent physical steps.

The first standing evaluator let initial standing trials advance into the
recording's later motion. The corrected evaluator keeps the current received
pose fixed for30 seconds, retains its received prefix, and uses stationary
reference starts. Earlier standing results are superseded.

Snapshot checks show zero observation, target and physical-continuation error
at six split points, including input-loss states. Fault/rearm state survives;
incompatible wrappers and duplicate observation timestamps are rejected.

The packaged launcher completed a 100-control wiring run during training.
One control deadline miss and 13 late physics finishes were reported; the
deadline fault latched. This loaded run is not a timing qualification. The final
nominal repetitions ran separately with training and rendering stopped.

Sensor adapter tests pass, including initialization, real delayed samples,
deterministic noise and latched warmup/update failures. Sensor mode never imports
privileged simulator history, including during initialization.

## Controlled pilot

Both arms restart from `native_target_ppo_v2/actor_00185.pt` with unchanged
architecture, actor normalization, reward transform, motion-prior reward and
learning settings. The simulator repairs apply to both arms. Only controller
memory differs between A and B.

Two seeds;128 worlds;64 controls/update;20 critic-only warmup updates followed
by at most40 actor updates per arm. Four-second movement snippets and existing
reset-start proportions remain. Per-world reset RNGs preserve paired sampled
physical starts even when worlds finish in different orders. Walk008 is held out.

All learned actors are evaluated with restored controller memory. Frozen
baseline is also tested with legacy memory to distinguish an immediate history
effect from learning. Selection uses fixed physical trials, not training loss.
Continuation requires more movement passes in B on both seeds, with no physical
completion or standing-success regression. A failed comparison ends this recipe.

Corrected result: each frozen/trained candidate passes only 1/192 fixed movement
trials. Seed0 physical completions rise from281 to284; seed1 rises from282 to287
(320 trials per arm). Every arm completes all64 standing trials physically,
but none passes the quiet-standing limits sampled at50 Hz. The final full-motion
clock separately scores continuous quiet standing at500 Hz. Short-trial standing
scores are not full replay qualification.

Neither seed gains a tracking pass. Checkpoints100/200 are therefore not run,
no candidate is promoted, and this training recipe is stopped. Snapshot, solver
memory and adapter correctness repairs remain independently justified.

## Independent-clock replay result

Training and rendering were stopped before these runs. Each requested the full
walk002 recording plus30 seconds of standing, with the same pinned wrapper.

| Run | Physical duration | Control misses | Late physics finishes | Controller p95 / max ms |
|---|---:|---:|---:|---:|
| Retained actor185, repeat1 | 13.592 s | 0 | 0 | 4.811 /15.961 |
| Retained actor185, repeat2 | 58.340 s | 1 | 3 | 4.829 /16.127 |
| Retained actor185, repeat3 | 19.244 s | 0 | 0 | 4.447 /13.983 |
| Trained restored-memory seed0 | 12.170 s | 0 | 0 | 4.286 /9.805 |

All tracking and continuous-standing verdicts fail. Three early stops violate
native joint position bounds. Repeat2 keeps physics running, but a deadline fault
at0.441789 s stops input admission after34 of1428 packets. Its58.34-second runtime
does **not** demonstrate completing the motion. The earlier successful walk002
motion/hold is not reproduced in this three-run current-wrapper sequence.

The comparison video includes the entire58.34-second requested timeline, all
three retained runs, the fixed seed0 candidate and original hand/head goals.
It explicitly marks input faults and physical stops. Stopped controllers have no
invented continuation. `comparison_video_v1` is an unfinished superseded render;
`comparison_video_v2` is the complete deliverable, verified through its last
video timestamp at58.340 s. No successful motion/hold is claimed.

Measurements: `controller_state_ab_v1/final_rollouts_v1/summary.json`.
Decision: `controller_state_ab_v1/corrected_pilot/decision.json`.

[Full comparison video](E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/controller_state_ab_v1/comparison_video_v2/native23_controller_memory_comparison.mp4)

[Complete result and pinned configuration](E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/controller_state_ab_v1/RESULT.json)

## Run the retained simulation

```powershell
& Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof\artifacts\onboard_inspection_20260912\RUN_CURRENT_NATIVE23_TELEOP_SIM.ps1
```

Default: walk002, recorded poses, ground-truth simulation feedback, unchanged
actor185, original native actuation, bounded preview and standing capture.
`-Clip` also accepts walk003, pico and walk008. `-Render` renders the saved trace
after simulation. `-Actor` selects an explicitly compatible experimental export.
No robot transport or motor publisher is present in this launcher.

`-Feedback estimated` selects the sensor adapter. It deliberately rejects the
existing airborne replay start as unsupported stationary initialization. A
qualified supported-standing initialization sequence is still needed before
this stage can run end to end. Delayed sensors require real warmup samples;
the adapter does not synthesize history.

Live input uses `-InputMode live`; the existing producer must use
`--native23-body-packets`. `-RearmFile` names an operator-controlled file whose
modification requests explicit rearm. Initial live launch arms one session;
faults never rearm automatically. `live_packets.jsonl` stores complete received
packets. `-PacketReplay` replays that file with its adjacent original `trace.npz`;
original source timestamps remain unchanged and clock rebasing is explicit.
These integration paths are implemented but are not live-qualified.

## Remaining stage gates

Full recordings and30-second quiet holds under original full-body/physical
limits; perturbation and early/mid/late input-loss recovery; successful repeated
full-duration timing without concurrent training/rendering; sensor-feedback qualification;
then calibrated live Pico driving simulation. Motor tests remain a separate
supervised stage. No robot settings, services or other laptop connection were
changed during this implementation cycle.
