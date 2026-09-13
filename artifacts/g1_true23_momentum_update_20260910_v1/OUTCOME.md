# Momentum update: modest prefix improvement, full candidate rejected

Native23 full-body deployment remains NOT READY. No robot commands, promoted
policy, changed physical gain/effort/range or fabricated source completion.

The internal-source assimilation diagnostic exactly reproduces1630 prior
forecasts. Copying native root/retained state causes a missing-momentum mismatch;
waist-roll equivalent correction p951.29845rad/s, maximum5.60854rad/s. This
supports testing momentum-consistent assimilation, not claiming its sufficiency.
All four full matched-source preflights reproduce missing q/dq exactly, with
identity updates. Source-only prediction equality is not teacher qualification.

The one native full request completes1980/6530 controls,1630/5780 source
controls=32.60/115.60s, versus25.58s before this change. At next control1980,
the unchanged native inward range search rejects; missing virtual bounds had
not failed. Actual native range/effort excess0 and velocity ratio0.571542.
Independent saved-target integration reproduces19,800 substeps bit-exact;
independent history/assimilation audit and96 frozen-source reinferences exact.

Equal25.58s source prefix, momentum/prior-copy/zero-model:

- Leg RMSE:0.119676/0.124231/0.124182rad.
- Root p95:0.218382/0.228086/0.813480m.
- Arms RMSE:0.263045/0.261937/0.295870rad.
- Momentum relative ankle p95:0.098996/0.115631m, still poor.

Whole32.60s executed prefix worsens to root p950.504244m, leg RMSE0.137673rad.
No full source, standing return, real-time, headset or hardware pass.

The failed state is right ankle roll0.233984rad at+2.56658rad/s,27.816mrad from
its upper limit. All8 old inward candidates reproduce failure. Separate100ms
constant-target diagnostics flag risk two controls earlier, but are not a
prediction of the future feedback controller. No stronger guard was installed.

Offline coordinated target search finds ONE feasible20ms continuation by
changing right hip pitch/ankle pitch/roll, but448 predictions take500.5ms and
change targets up to2.38743rad. Segment refinement still requires2.15544rad.
Rejected as a live fix: no closed-loop controller trial, slew qualification or
timing pass. Optimizer convergence itself was false; final constraint feasibility
was independently checked. No hard-coded replay-specific correction adopted.

Tests:20 momentum/history focused tests pass;28 combined momentum/coordinated
tests pass; scoped Ruff E/F pass. Source and exact commands live in EXPERIMENT.md,
run_native.py, verify_predictor.py, audit_native.py and the stop diagnostic
drivers. They refuse existing output directories: do not rerun in these saved
paths. Use a separately versioned output location for any authorized experiment.

Evidence SHA256:

- Native report:2294d1381ca34ae9470731d27ecdfd7e74c60c19e37f8256ff97e3836df78f6b.
- Physics audit:f388b7e8e9edad685ef0745292f05c88d8356121aa1502249c04cea1cd0d4c75.
- History audit:b1d887ce202f6c85ff7b56b5c7bc027f64b5e82643ff10acaab226a7b1a519bd.

The subsequent current-observer variant has separate artifacts and is also
rejected. Do not silently switch this model into the live receiver.
