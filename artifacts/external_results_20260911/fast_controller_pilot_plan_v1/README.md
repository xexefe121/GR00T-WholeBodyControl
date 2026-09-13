One selected experiment: qualified walk003 nominal imitation around frozen BFM,
followed by at most one evidence-gated DAgger round. Initial nominal pilot is
authorized and implemented separately in `fast_controller_nominal_pilot_v1`.
DAgger has not been authorized to run.

The old direct student already had an uncapped linear output and failed before
source motion despite 0.027 rad teacher-state error. The old BFM residual had a
real 0.25 rad cap: it cannot copy the new walk003 teacher. Exact saturation
canonicalization changed only 45/18,837 source target components and reduced
target slew RMS by 0.49%; it does not remove the demonstrated range problem.
This pilot therefore removes that ceiling and measures actual closed-loop
behavior. It does not assume command range is the only failure cause.

Inputs remain 1,069 floats: measured native proprioception and previous target,
eight full-body goal samples at offsets 0,1,2,4,8,16,24,37; frozen BFM's current
unclipped target; and previous combined action. Student goals use the declared
v4 native reference plus immutable original29 hand/head intent. The BFM prior
keeps original native goals, horizon 8, position gain 1 and yaw gain 2. Neither
clip ID, source frame, clock, MPC state nor saved gain enters the network.
The harness uses frame indices only to deliver packets at the original clock.
This remains offline simulation with root pose/velocity and 0.74 seconds of
received goal packets, up to 0.76 seconds of raw-pose derivative support.

The network is 1069→256→256→23 with ELU hidden layers and a zero-initialized
linear final layer. Output is multiplied by each native joint's span; there is
no tanh or residual radius. The final physical target alone is clipped to the
unchanged native joint target bounds. Base target arithmetic stays exactly
`default + raw_action*.25*training_effort/kp`. Combined preclip action becomes
the next student/BFM history input. The zero head must reproduce the same
frozen ONNX BFM prior exactly before the first optimizer step.

The demonstration is the physically and geometrically qualified
`walk003_terminal_bfm_yaw4_hybrid_v1`. Its first 1,269 controls retain actual MPC
entry/acquisition, all 819 source controls, and return. Labels are actual
applied native PD targets minus the BFM unclipped target evaluated on the
teacher's exact saved precontrol state/history. Teacher prefix histories use
normalized actually applied targets, not unavailable or arbitrary preclip MPC
commands. Initial entry samples are not described as settled standing.

Fit once: seed 773, 1,000 AdamW steps, learning rate 3e-4, weight decay 1e-5,
batch 256 with 64 samples from each entry/acquisition/source/return stratum,
feature standard-deviation floor 0.05, gradient norm cap 10, one CPU thread.
All 1,269 labels train; teacher-state error is a training diagnostic, not heldout
validation. No cap, epoch, seed or architecture sweep. Export the head to ONNX
and check its numerical agreement with Torch.

Evaluate one uninterrupted native MuJoCo 3.2.3 lifecycle from reference frame
10: 1,569 controls, 500 Hz native PD. On the declared returned-standing event
at control 1,269, bypass the learned head and keep the already qualified frozen
BFM yaw-4 terminal controller. Preserve current measured history at the seam.
After a complete lifecycle, continue another 250 controls in a separate trace
without resetting physics/history. If the lifecycle fails, retain the failure
and do not skip directly to the extension. Check actual joint positions,
velocities, actuator forces, warnings, quaternion validity and independently
accumulated time at every 2 ms. Score original root/yaw/hand/head intent and v4
legs/feet, plus unchanged quiet-standing diagnostics. Report inference p50,
p95, maximum and 20 ms deadline misses separately from source qualification.

Minimal implementation already added under distinct artifact-local names:
`student_linear_runtime.py`, `collect_nominal_labels.py`, `fit_linear_head.py`,
`initialize_zero_head.py`, and `evaluate_nominal_pilot.py`. Existing GoalFeatures,
guarded ONNX BFM and quiet metrics were frozen and reused. Existing student
evaluators were not suitable unchanged: their physical failure helper allowed
0.01 rad excess, used reassociated base arithmetic, lacked the required full
clock/warning ledger, and lacked the qualified terminal switch. No shared MPC,
model, gain or actuator-cap code changed. Source version 1 is preserved;
reviewed version 2 adds reliable failure finalization and pre-fit parity staging.

If this nominal pilot motivates one DAgger round, review the oracle interface
first. The evaluator retains full MjData integration state plus measured
history/prior action at every actual precontrol state, including the first
failure/query point. Select one documented actual student snapshot; never
fabricate perturbation labels or extrapolate stored feedback K. Freshly replan
the original remaining walk003 sequence from that exact snapshot, using its
declared teacher objective and native physics. Validate the whole remaining
source/return and terminal hold independently before admitting any correction
labels. A feasible H30 segment alone is insufficient: recent recovery evidence
shows short-horizon cost reduction can destroy BFM continuation feasibility.
Reject and retain an unsuccessful oracle branch; do not cycle through query
points to manufacture a valid dataset. If admitted, aggregate that one actual
expert branch with the original demonstration, refit once with the same fixed
settings and repeat the full nominal lifecycle/hold once. A nominal success
would still not establish robustness, heldout-motion performance or hardware
observability.

Measured compute anchors: original 1,000-step CPU residual fit took 6.7 seconds;
full 1,569 teacher-state feature audit took 27.5 seconds; old walk003 MPC full
lifecycle took 1,951 seconds. Initial pilot should need roughly 1–3 minutes of
actual collection/fit/evaluation compute after implementation, subject to CPU
contention. A qualified full-remaining DAgger oracle may cost up to roughly
30–45 minutes for one early query; it is the expensive, conditional step.
These are estimates, not realtime claims. No PICO data, devices or training are
part of this pilot.
