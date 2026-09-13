Prepared plan only. No MPC teacher labels or student weights have been created.
Proceeding depends on the real MPC probe results, not reference FK quality.

Use actual committed controls from a physically executed MPC rollout:
`qpos[k]`, `qvel[k]`, `source_frame[k]`, previous executed target, and
`target[k]`. The saved qpos/qvel have T+1 rows; targets have T rows.
The label pairs the pre-action state at k with the executed target at k,
including the MPC feedback correction and native target clipping. Preserve
original source files, frame IDs, timing, physical limits and failure outcomes.

Current saved `planned_state`, `planned_target`, and `feedback_gain` describe
the local planner/controller. They are useful for inspecting local feedback.
The last backward-pass K may have been computed before the final line-search
trajectory or a rejected sweep. Do not present arbitrary K-perturbed states as
new physically validated MPC expert examples. Additional training states should
come from simulated perturbed rollouts with the real teacher queried there.

Minimal student: a small two-hidden-layer MLP predicting 23 normalized joint
targets, with the same native clipping, BFM PD gains and full native effort caps
used to execute the teacher. No frozen BFM forward pass is required. Use simple
supervised target regression initially; inspect per-joint errors and saturation,
then assess complete closed-loop physics with the student supplying every
control. Low regression loss alone does not establish balance.

Input contract must account for preview. The current MPC uses a 15-knot,
300-ms source preview. A current-goal-only student cannot be expected to
reproduce that anticipatory controller exactly. The clean initial comparison
uses the same declared 300-ms received-reference buffer for both teacher and
student: current physical proprioception75 plus native/original task goal90
at all15 knots (1425 floats total). A 256x256 MLP is still small. Later reduce
preview or inputs only as a separately measured change; do not silently claim
the 140-ms stream contract. No simulation state, source frame removal or source
time warping is involved.

Proprioception comprises current joint position/velocity, base angular velocity,
projected gravity and previous applied target. Goals retain native joint/feet/root
references and original29 VR21/task intent, expressed consistently relative to
the current physical root. Fit normalization on training recordings only.

First completed three-source-second MPC probe supports a limited student
plumbing experiment for that segment. Extend teacher collection to full clips
and recovery/perturbed states before claiming broad tracking. Split by complete
recording/rollout, rather than randomly interleaving adjacent frames. Report the
previously seen development split honestly. Evaluate original29 hands/head,
native feet/root, full source duration, actual joint ranges/speeds/efforts, and
policy latency on every subsequent student replay. Any failed teacher rollout
remains failed evidence, not a qualified expert demonstration.
