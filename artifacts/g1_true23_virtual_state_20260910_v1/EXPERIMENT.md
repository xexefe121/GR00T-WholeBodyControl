# Missing-axis internal-state consistency hypothesis — SIM research only

The previous discarded-action-memory trial retained six source commands while
forcing those joints' position and velocity inputs to zero. It failed earlier.
That does not test a self-consistent, explicitly hypothetical source-joint model.
This is not the original SONIC-transfer analytic codec or a claim that this
23-DOF robot has six more sensors/motors. Physical state must remain23-DOF.

First test predictability, without native23 dynamics or policy training. Fit one
affine discrete-time model from the existing full original29 PICO-derived
simulator trace. Outputs are the next six absent position offsets and velocities.
Inputs: current internal12 state, current source29 command, current retained23
position/velocity, root gyro and gravity. All inputs exist causally at inference.
Use one fixed standardized ridge fit, lambda0.001; no hyperparameter sweep.
The learned object is an internal source-dynamics approximation, not a new
tracking policy or a qualified full-motion teacher. Source controller tracking
and physical-limit failures remain disclosed in the original reports.

Evaluate zero-initialized free-running state prediction on the entire existing
PICO trace and on three complete walking source29 traces excluded from this fit.
These walking recordings were used previously for other development, so they
are not pristine held-out data. No reference cropping, retiming or reset midway.
Check the fitted internal state's discrete eigenvalues; an unstable recurrence
cannot progress to a native23 trial. Keep every failed result.

On96 uniformly spaced original inputs per clip, compare frozen normal29 decoder
outputs with: all absent channels zero; absent actions retained but q/dq zero;
predicted q/dq and the same historical absent source actions. Encoder and all
retained measured channels remain unchanged. The predictor is driven by saved
source actions for this diagnostic, not its changed decoder outputs. This is
an open-loop consistency test, not closed-loop/native23 tracking success.

Predeclared development gate for one later native23 experiment: spectral radius
<1; on every fit-excluded walk, missing position RMS at most75% of zero-fill RMS,
velocity RMS no worse than zero-fill; retained-leg output RMS closer to the
original source decoder than BOTH zero-fill and action-memory-only variants.
These are hypothesis-screen thresholds, not relaxed deployment criteria. Even
a pass only justifies a guarded full native23 trial using the same physical
model, limits, timing, source and original normal SONIC weights. No physical
robot commands, headsets, new motion downloads or native-state fabrication.
