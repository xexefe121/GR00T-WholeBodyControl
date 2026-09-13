# Current-observer timing hypothesis rejected

The new before-inference momentum update performs worse than after-inference
assimilation on identical source frames. No deployment or controller promotion.

Four complete source-only preflights reproduce all missing q/dq exactly. New
benchmark measurement hook passes bit-exact old/new five-control physics and
copy-isolation tests. Thirteen current/momentum tests pass. No old executed
benchmark, source, native gains/efforts, range guard or training weights changed.

One native full request:2002/6530 controls,1652/5780 source controls=
33.04/115.60s. Internal predicted missing-axis range excess0.01790235rad rejects
the candidate. Maximum internal missing-velocity correction6.75871rad/s. Actual
native velocity ratio0.755723. Independent saved-target replay of20,020 substeps
matches qpos/qvel exactly; all2003 history/forecast/assimilation records and
96 fresh frozen-source reinferences match exactly. Code equality does not make
the failed motion succeed.

Common25.58s source prefix, current / after-inference momentum:

- Root p95:0.310110 /0.218382m.
- Leg RMSE:0.120386 /0.119676rad.
- Relative left/right ankle p95:0.099711/0.117093 /0.098996/0.115631m.
- Arms RMSE:0.261343 /0.263045rad.

Entire available33.04s prefix: root p950.758982m, legs0.145735rad, relative
ankles0.145237/0.140384m. Full source and standing return not reached. No timing,
contact/slip, live headset, source-pose acquisition or hardware handback proof.
Reject this hypothesis; no more virtual-state variations without new evidence.

Files: run_native.py and verify_predictor.py bind full request and source
preflights; audit_native.py independently reconstructs histories/outputs and
compares four policies over a shared prefix. Existing output directories are
immutable: do not rerun these drivers against the saved directory.

Evidence SHA256:

- Native report:beb3e11b4706b382ee36ee269a37007652b9160bcdc145b6b64ab3e837ea7b82.
- Physics audit:38f9f6b7f2d90522671248a7014f4daec94a4c6279d594a3557b980e5e38a93e.
- History audit:5dfb50eb6c72e47036a10f8ae412973ed04a4072dfd256d789eadda1246a1512.

All experiment/audit processes terminal. Full-body goal stays ACTIVE; actual
leg/foot tracking remains unresolved. No physical robot commands were issued.
