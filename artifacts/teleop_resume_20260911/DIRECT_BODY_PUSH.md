# Direct received-body goal route

Implemented within the original six-hour window, starting06:54UTC.
Original causal800, balanced400 and neutral-adapter single-policy400 recipes
remain finished and rejected. No unchanged extension.

New g1_true23_direct_body_goal.py adds current native body pose inputs to the
existing pretrained BFM backward encoder. Input1749 = existing1323 full-body
task/state/history features + current425 BFM reference features + one causal
goal blend. Every joint/body velocity is recomputed from received past.
The same actor handles all phases; goal blending preserves initial neutral
standing and introduces actual body poses. There is no target-policy handoff.
All BFM actor tensors remain trainable, with the same full native target range.
Original hand/head task features and all body objectives remain intact.

CausalController recognizes input width1749 and reuses CausalReceiver's owned
packets. It computes the new current-body inputs from its last two samples,
and advances goal blending once per control command. Training uses exactly
the same causal reference derivatives and delayed blend update. Reference
kinematics were prepared with native MuJoCo3.2.3 into the existing bank's
bfm_reference_inputs_v1.npz. No held-out walk008 training rows are added.

Input check:12 packet states across all four clips match training body inputs
exactly. Mutating a future sample has no effect. Zero goal blend preserves
the previous neutral controller exactly in the checked state; full body
conditioning changes target output by.687rad, establishing an active input
route. These checks are not physical qualification.

direct_body_smoke_v1 is a4-update throughput smoke, paused at initial export
for native standing and all-four full baseline attempts. Session24374.
Initial standing report direct_body_initial_standing_v1 passes30s at0,+.03,
-.03m/s. It is unpaced, and includes a23.8ms maximum policy call.
Initial full baseline direct_body_initial_full_v1 is still evaluating:
walk003 completes61.38s physically with both quiet windows, tracking fails;
walk002 hits a native joint bound at24.592s, after all667 source samples.
Pico and walk008 results are pending at this checkpoint. Session62717.

Do not release smoke PPO until the full initial baseline finishes. If laptop
throughput and remaining time permit, one main pilot may run up to200 total
updates including smoke, with two full evaluations and an absolute end before
08:46:40UTC. Preserve current snapshots before main training. No duplicate
jobs, paid compute, real Pico, DDS or robot commands. All native/tracking/
standing/timing/disturbance/input-loss gates remain unchanged.

Initial full baseline finished07:05UTC. All four tracking gates fail.
walk003 completes61.38s plusquiet, rootp95 1.792m/legRMSE.309rad.
Pico completes160.60s plusquiet, rootp95 .283m/legRMSE.218rad/head.089m;
hands and feet still outside limits. walk002 joint-bound failure24.592s,
walk008 joint-bound failure12.948s. Thus no candidate selected. Source-body
conditioning improves some joint/head measures but does not establish overall
tracking improvement or physical qualification. CONTINUE released4 smoke
updates only after these native initial results and standing passes.

Smoke finished4 updates,1024 transitions/update in2.85..3.46s. Smoke4 retains
all three native30s standing passes; walk003 completes61.38s with both quiet
windows but tracking still fails (root1.800m,heading57.24deg,leg.303rad).
Main pilot_direct_body_v1 started07:09UTC, released07:11UTC. Initial ONNX is
byte-identical to tested smoke4.196 more updates, global200 maximum;64x64
rollouts,16 minibatches, actor initial learning rate1e-8, no bootstrap,
optimizer fresh. Wall cap.9h including initialization, before08:46:40UTC.
Session30207 owns the only ML job. Current sources saved in its source_snapshot.
Evaluate actor_00098.onnx (global102) and actor_00196.onnx (global200) once
each on all four clips plus30s hold, WITHOUT --balanced. Do not extend.
First3 main iterations measure678..704 transitions/s,5.62..6.04s per4096
transitions. Global102 and200 evaluations are already queued in sessions63960
and98864 respectively. Each waits for its completed normalization archive,
then runs all four clips plus30s hold. Outputs direct_body_global102_eval_v1
and direct_body_global200_eval_v1. Do not launch duplicates.
