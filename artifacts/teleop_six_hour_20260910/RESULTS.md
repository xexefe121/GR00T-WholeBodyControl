# Native 23-DOF G1 U2 simulation work

**Six-hour work window closed, 2026-09-10 19:22 UTC. All experiments ended.**

Full-body teleoperation is not yet qualified. The strongest completed result
is two full nominal walking recordings on the actual 23-actuator model, with
all recorded-source tracking and physical gates passing. A separate terminal
standing-policy switch also passes quiet-standing checks. The latest PICO
candidate failed a native ankle limit at 68.17 source seconds. No physical
robot or Pico device was operated.

## What is working

- Native MuJoCo 3.2.3 model: 12 leg joints, one waist joint, five joints per
  arm. Physics runs at 500 Hz, control at 50 Hz, with native PD, effort,
  speed and joint bounds. No state copying or root forces after initialization.
- The supplied [mjbatch project](https://github.com/kevinzakka/mjbatch) is
  integrated as an offline planner on this native23 model. Source revision,
  compatibility patch, model, physics and input hashes are recorded.
- A fresh native BFM policy rollout from actual measured state and history
  provides an additional MPC initial guess. The planner evaluates all
  candidate guesses under the same physical rollout and objective.
- Independent verification checks every 2 ms physical state and torque,
  original source intent, complete source timing, simulator warnings and
  lifecycle completion. Failed and incomplete candidates remain explicit.

## Completed evidence

| Case | Complete source / lifecycle | Measured result |
|---|---|---|
| walk003, stronger joint margin | 819/819 source controls; all 31.38 s | All 15 recorded-source gates pass. Independent WSL physical replay matches every state and torque exactly. Separate BFM standing switch also passes source and settling checks. |
| walk008, stronger joint margin + relative-foot cost | 364/364; all 22.28 s | All 15 recorded-source gates pass. Independent WSL replay matches every state and torque exactly; final 3-second quiet-standing checks pass. |
| PICO, fresh BFM seed + stronger margin + relative-foot cost | 3408 complete source controls plus one half control, of 5780 requested; 68.17 source seconds | Rejected. Left ankle pitch exceeds its lower native bound by 0.0106374 rad. Aggregate tracking thresholds pass on the measured trace, but physical safety and full completion fail. |
| PICO, earlier fresh BFM seed with default margin | 1415/5780; stopped at 28.3 source seconds | All measured tracking thresholds pass, but actual left ankle exceeds its bound by 0.00356449 rad. Rejected and incomplete. |
| walk002, original-reference MPC | 667/667; all 28.34 s | Left relative-foot p95 0.13685 m and ankle excess 0.00182487 rad fail. Other tested variants also failed. |

For the passing walk003, original root p95 is 0.09046 m, heading p95 9.032
degrees, all-leg RMSE 0.08738 rad, relative feet p95 0.10097/0.09692 m,
original relative hands 0.05270/0.05802 m and head 0.03913 m. Every physical
joint bound, speed and effort check passes; simulator warnings are zero.

The full video preserves all source motion, entry, return and standing:
[walk003 complete replay](E:/codex-artifacts/sonic23_teleop_six_hour_20260910/visual_walk003_allmargin_nominal_full_v1/full_source_and_lifecycle.fixed_world.mp4).
Its [independent gate audit](E:/codex-artifacts/sonic23_teleop_six_hour_20260910/bfm_online_intent_v2/walk003_allmargin_wsl_independent_replay_v1/recorded_source_audit_v2.json)
is the qualification evidence. The fixed world camera and original hand/head
markers expose trajectory drift rather than aligning it away.

For walk008, original root p95 is 0.06616 m, heading 3.434 degrees, leg RMSE
0.08961 rad, relative feet 0.05636/0.06981 m, original relative hands
0.06241/0.07943 m and head 0.03984 m. The earlier right-foot failure is fixed
in this completed cost variant. Its
[full gate audit](E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_full_v1/walk008_v4_native323_allmargin_relativefoot_full_v1/recorded_source_audit_v2.json)
and [independent physical replay](E:/codex-artifacts/sonic23_teleop_six_hour_20260910/bfm_online_intent_v2/walk008_allmargin_relativefoot_wsl_independent_replay_v1/report.json)
bind the complete result. No single identical controller configuration has
yet passed all four sources.

The final PICO result retained all 37,585 physical steps. The failure is left
ankle pitch at source 68.170 s: actual angle -0.8833074 rad versus native lower
bound -0.87267 rad. Its target was already 0.0524 rad inside that bound, so
simply shrinking target limits is not an established fix. There were no falls,
simulator warnings, speed violations or effort violations. The original root
p95 is 0.16281 m, heading 3.193 degrees, leg RMSE 0.07232 rad, relative feet
0.07139/0.07022 m, original relative hands 0.04945/0.08404 m and head 0.03408 m.
These aggregate metrics include the terminal partial sample; they do not
make the failed 68.17-second prefix a completed 115.6-second recording.
The [final strict audit](E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_full_v1/pico_v4_native323_freshseed_allmargin_5iter_full_v1/recorded_source_audit_v2.json)
exits with failure. The earlier 57-second checkpoint passed every measured
threshold and remains available as explicitly incomplete evidence.

The final independent diagnosis reproduced the saved physical prefix exactly.
The accepted nominal 20 ms endpoint already predicted a 0.02404 rad ankle
violation; privately completing the failed control produced 0.02377 rad.
This was not an excursion hidden between otherwise safe sampled endpoints.
The optimizer accepted a predicted violation despite its soft margin. The
next repair needs explicit plan feasibility enforcement and a verified
fallback, rather than another unverified increase of the soft penalty.
The [failure diagnosis](E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_full_v1/pico_repair_final_control_diagnosis_v1/report.json)
keeps that private continuation separate from the unchanged failed trial.

## What remains unresolved

The original walk003 MPC tail had residual standing motion: last-second peak
joint speed 7.834 rad/s and final maximum joint speed 0.493 rad/s. A separate
hybrid preserves every source and return target, then switches to native BFM
at the final standing phase. With terminal yaw correction gain4, it passes
the unchanged source gates and the separately declared quiet-standing checks
for the original lifecycle and an additional 5-second hold. Final lifecycle
joint speed is 0.00252 rad/s; final-3-second joint speed p95 is 0.00753 rad/s,
root linear speed p95 0.000435 m/s and original heading error p95 3.094 degrees.
The additional hold has heading p95 3.231 degrees and joint speed p95
0.00166 rad/s. This terminal hybrid is a separate tested controller variant,
not a change silently applied to earlier results.

Saved-feedback replay on Windows diverged despite the same MuJoCo version;
robustness is not established by the nominal WSL replay. The standing change
does not solve that earlier moving-source feedback failure.

The MPC result is offline. It uses at least 740 ms of reference packets and
up to 760 ms of raw pose support. The passing walk's planning p95 was 10.09
seconds per100 ms control block, with every block missing its deadline. The
latest five-iteration PICO planner was faster but still offline: median2.338 s,
p95 3.512 s and maximum4.084 s per100 ms planning block; all752 deadlines
were missed.

A separate received-only native BFM stream completed 1,824 paced controls
with no missed 20 ms deadline on a quiet machine: loop p95 9.616 ms, maximum
16.451 ms. It still fails full-body tracking. That timing success and the
offline MPC tracking success are different controllers; no combined live
success is claimed. Loaded-machine timing and sensor-only estimation also
remain unresolved.

The existing residual student with a 0.25 rad correction cap cannot reproduce
the passing MPC teacher's chosen targets. Even a perfect capped head leaves
0.24881 rad source command RMSE. This is a command representation result,
not proof that every alternative controller with that cap is impossible.

Recorded full-body optical input is not proof that live Pico trackers can
reconstruct all leg motion. Live Pico work remains a later stage after a
single controller passes complete, robust simulation and timing checks.

## Reproduce and continue

[SIM_RUNBOOK.md](Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/SIM_RUNBOOK.md)
contains commands for the passing independent walking replay and the strict
auditor. [RUN_PASSING_WALK.ps1](Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK.ps1)
was tested end to end and reproduces the complete walk with quiet terminal
standing and the separate5-second hold. The
[complete36.38-second hybrid video](E:/codex-artifacts/sonic23_teleop_six_hour_20260910/visual_walk003_terminal_bfm_yaw4_full_v1/full_lifecycle_and_separate_hold.fixed_world.mp4)
was independently decoded and visually checked.
[Current PICO planner command](Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_v1/REPRODUCE_CURRENT_PICO.md)
records the exact offline candidate invocation.
[SIM_ACCEPTANCE.md](Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/SIM_ACCEPTANCE.md)
contains the predeclared gates. [SESSION.md](Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/SESSION.md)
records the complete experiment handoff, including rejected alternatives.

Next decisions must use the final long-trial outcomes: resolve the late PICO
ankle failure and validate standing transitions at arbitrary stops, then test one consistent
controller on all four complete sources and perturbations. Real-time tracking
requires a controller that meets the actual 20 ms deadline; more offline MPC
success alone does not resolve that gap.
