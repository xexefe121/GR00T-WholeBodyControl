# Native preview controller

Status: walk002 completes its 58.34-second lifecycle and 30 seconds of quiet
standing with independently advancing physics. All packets are consumed; no
control deadlines or observation publications are missed. One physics finish
is 2.9163 ms late. Full-body tracking fails, so this is not simulation ready.
The same-controller suite finished. Walk003 stopped at14.912s on the right ankle
roll bound; Pico at63.660s and walk008 at10.972s on the right shoulder roll bound.
All four recordings fail full-body tracking. Current state-restoration work is
documented in `CONTROLLER_MEMORY_EXPERIMENT.md`.

## Implemented controller

The actor is the unchanged `native_target_ppo_v2/actor_00185.onnx`. Its 23
targets pass through a measured-state native-PD preview. The preview keeps
the original gains, effort caps, target bounds and physical stopping criteria.
It adjusts currently limiting target coordinates, with at most seven candidate
queries. It receives no future motion, clip identifier or prepared motion plan.

The independent implementation checks immediate application and a 12 ms delay,
holding the actual previous target during the delayed part. Both scenarios
cover a 32 ms horizon and run on separate physical CPU cores. Native and Python
preview outputs match exactly on the sampled physical states. This is a bounded
prediction method, not a recursive safety guarantee.

After received non-standing leg motion returns to a standing leg pose, the
controller can stop the factory gait clock. It requires 200 ms of stationary
root/leg/foot references and measured root position, yaw, tilt and velocity
conditions. Initial acquisition stays unchanged. All leg, foot, arm and head
objectives still enter the learned actor; only the gait command stops. Packet
fault latching and explicit rearm remain unchanged.

The native clock uses versioned atomic payloads for observations and commands.
A preempted reader cannot reject publication. Readers reject a mixed snapshot;
missed observations and deadlines still fault normally. A separate-process
stress test published 20,000 messages on each of three channels without a
rejected write or mixed accepted payload. It does not qualify real-time timing.
The first thread-based test timed out in its Python fixture; the process test
matches the runtime architecture.

## Results so far

All paths below are relative to
`E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1`.

- `native_preview_guard_v1`: unpaced guard-only tests. Walk002 survives 58.34 s
  but fails the 30-second quiet window. Walk003 falls at 14.678 s; Pico reaches
  79.564 s and walk008 9.302 s before joint-bound stops. All tracking fails.
- `native_preview_capture_v1/walk002`: rejected standing-capture version;
  changed acquisition and stopped at 14.372 s.
- `native_preview_capture_v2/walk002`: corrected capture. Full 58.34 s and
  30-second quiet hold pass. Control p95/max: 5.512/9.927 ms, unpaced.
- `received_sim_runs/20260913_103814_831_walk002`: immediate-only preview on
  independent clocks stops at 12.882 s. Application delay is 4–8 ms despite
  zero missed deadlines; that delay was absent from the original preview.
- `received_sim_runs/20260913_105322_932_walk002`: delay-aware preview stays
  upright for 58.34 s, but a dropped observation causes a deadline fault at
  18.56 s. Normal packet consumption stops; this is not motion completion.
- `received_sim_runs/20260913_111006_874_walk002`: atomic clock and separate
  physical cores. All 29,170 physics steps complete, all 1,428 packets consumed,
  no input fault, and the 30-second quiet hold passes. Control p50/p95/max:
  2.982/3.859/9.094 ms. Application delay p50/p95/max: 4/6/10 ms. One physics
  finish is late; maximum lateness is 2.9163 ms. Full tracking fails: root p95
  1.058 m, feet p95 0.320/0.338 m, leg RMSE 0.298 rad. These remain far outside
  the original limits.

Current independent suite: `native_preview_clock_suite_v4`. Selected libraries:
`native_preview_backend_v2/libtrue23preview.so` and
`factory_clock_v4/libtrue23clock.so`. Earlier libraries and the pinned Pico
demonstration retain their settings.

## Run

Use `RUN_RECEIVED_NATIVE23_SIM.ps1` with an explicit native actor and
`-TaskCommands -NativeTargets -NativePreviewGuard -NativeStandingCapture`.
The independent candidate additionally uses `-IndependentClock`,
`-NativePreviewDelaySubsteps 6`, the preview library above through
`-NativePreviewLibrary`, and the atomic clock through `-NativeClockLibrary`.
No training extension, robot connection, motor publication or paid compute is
part of this controller experiment.
