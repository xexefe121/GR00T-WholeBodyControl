# Momentum-consistent internal observer hypothesis

Previous native23 experiment is REJECTED, preserved without edits. It improves
root/arm tracking on a25.58s common prefix, but virtual waist-roll oscillates
to11.43rad/s and crosses its bound; leg error remains unchanged. No native motor
range/effort exceeded, and no physical robot was operated.

The previous source-model measurement update overwrites root/retained q/dq,
leaving missing velocities unchanged. Coupled inertia means this can introduce
a generalized momentum jump on the missing axes. First measure this on all
1630 attempted saved-native controls and matched original source controls.
The diagnostic proposal does not alter old predictions; require old saved
predictions to reproduce exactly. Treat nontrivial correction (>0.05rad/s
95th-percentile absolute virtual waist-roll velocity) with zero matched-source
correction as grounds for a corrective SIM test, not proof of causation.

Proposed update preserves previous missing canonical momentum while matching
current root/retained measurements:
M_new_mm * v_new_m = (M_old * v_old)_m - M_new_mo * v_measured_o.
Missing positions remain untouched. Use scratch mass-matrix data, preserving
the prediction integrator caches. No tuning parameter, velocity/position clamp,
torque increase, virtual reset, new sensor, training or missing-truth input.
The correction is an explicit internal observer update, NOT native state editing.
Like the prior observer, current policy consumes the preceding model prediction;
measurement assimilation occurs after current inference, before next prediction.
This is not a physical conservation claim across two different robot morphologies.

Before a native test, require complete source-only preflight on the existing
PICO and three walking traces: missing q/dq <=1e-5 error and missing range excess
<=1e-6. Matched observations take the exact identity path. Also test the momentum
equation, state isolation, identity and invalid-input terminal behavior.

If those checks pass, run ONE full115.60s PICO plus unchanged standing lifecycle
native23 trial. Actual23 physics, limits, source references, target/range guard,
50Hz policy/500Hz physics and920ms buffer stay unchanged. Preserve any failure.
Compare complete/source-prefix legs, feet, arms and root to both zero-model and
rejected velocity-copy baselines. Survival or reduced virtual oscillation alone
is not tracking readiness. No hardware command or deployment promotion.
