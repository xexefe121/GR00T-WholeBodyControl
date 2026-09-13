# Unused ankle actuator authority — fixed copied-state diagnostic only

The completed100ms target-preview trial improves final1000 PICO from30.60 to
31.90 source seconds and restores a complete003 lifecycle, but all four motion
screens fail and interventions take longer than20ms. It is REJECTED. There is
no automatic horizon sweep, training extension or deployment promotion.

The preceding exact force diagnostic showed that the original final1000
rejected state already requests the strongest bounded inward ankle-roll PD
target, but only14–16Nm of its35Nm rated torque is generated. Changing the
target search cannot access that remaining actuator authority at this state.

Test whether actuator authority, not just target authority, is locally usable.
Use exactly the previously rejected final1000 and parent500 PICO states/actions.
No resumed actual trajectory: reset independent probes from each saved current
q/v. Keep decoded target, original Kp/Kd, model, gains, native effort limits and
timing unchanged. Apply a constant right-ankle-roll inward feedforward request
of0,-2.5,-5,-10Nm in separate counterfactual probes, clipping TOTAL requested
actuator torque to its original effort limit at every2ms step. All other torque
requests remain the same PD law. No external spatial forces or extra actuators.

Record50 substeps/100ms. Evaluate the existing0.0019rad all-joint reserve over
the first10substeps/20ms, plus native velocity, total actuator effort and longer
constant-target consequences. Verify final1000 zero-offset first10states match
the prior saved rejected nominal probe exactly. Freeze this small4x2 matrix;
no chosen-offset feedback tuning or extension is authorized by this experiment.

This is a different controller-command hypothesis, not an accepted replacement
for the23-output policy or a loophole around its range filter. No result from
one copied state qualifies closed-loop VR, complete motion,50Hz runtime, live
estimation or physical recovery. Any later integration needs a separately
versioned, measured total-torque path and the unchanged physical checks. Do not
edit existing policy/controller sources or send any physical robot command.
