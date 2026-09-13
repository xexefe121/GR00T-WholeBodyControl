# Independent feasibility implementation review

Reviewed the opt-in native model predicates, iLQR core, tracker, runner and independent manual-PD referee. Shared code was read only; findings were fixed by their owners. No solver, MjModel, MjData or physics step was executed by this review. Pure-array, JSON/shape and floating-point clock witnesses are saved here.

The reviewed revision has no remaining blocker identified in candidate acceptance, accepted-gain retention or rejection handling. This is code-review evidence, not tracking, recovery, timing or hardware qualification. Exact reviewed files and hashes are in `reviewed_sources/` and `reviewed_source_hashes.json`; later edits are outside this snapshot.

## Findings and verified disposition

1. **Force arrays silently broadcast.** Initial `assess` accepted malformed `(N,1)` forces, so it could evaluate the wrong per-joint input. Pure-array witness reproduced this. Revised model requires exactly `(N,23)` force, `(N,8)` warnings and `(N,)` time. Wrong batch size is rejected too. Independently rechecked.
2. **Invalid initial root quaternion passed.** A zero quaternion passed finite/range/fall predicates. Revised model rejects norm error greater than1e-10, matching the oracle. Zero quaternion and norm1+1e-9 both fail the saved independent witness.
3. **Private preview did not enforce its unassisted torque-model claim.** A complete MjData copy also copies applied generalized/body forces. Revised helper rejects nonzero applied forces and requires unit-gain native torque actuation. Root oracle already enforced these restrictions. Fixed by inspection; this review did not rerun physical copy tests.
4. **Empty rejection outputs lost trailing dimensions.** Initial preview returned empty forces `(0,)`; zero-control runner rejection exported several arrays similarly. Revised outputs retain force/target `(0,23)`, planned state `(0,59)`, K `(0,23,58)`, and history `(0,300)`. Saved pure reporting witness checks these shapes.
5. **Actual-step predicates differed from declared hard predicates.** Original actual guard used clock tolerance1e-8 and comparisons that could miss NaN time/force. Revised opt-in actual loop calls the same predicate on every2ms post-step q/dq, actual generalized actuator force, warnings and independently expected clock. It stops immediately on rejection. Legacy opt-out behavior is preserved.
6. **Nonfinite failures could destroy the JSON report.** Raw NaN/Inf summaries conflicted with `allow_nan=False`. Revised hard-mode report and checkpoint metadata normalize nonfinite summaries to null while retaining explicit failure reasons. Raw arrays are not normalized. Saved pure witness confirms JSON validity and that original numerical inputs remain unmodified.
7. **Raw failed force/time evidence was missing.** Commanded torque alone does not retain a nonfinite actual actuator force; converting failed clock summaries to null loses raw time too. Revised opt-in trace stores actual `physics_actuator_force`, `physics_time`, `physics_expected_time`, `physics_warning_number` and `physics_warning_lastinfo`, appended before rejection. Force has N×23 samples; time/warnings include initial state and have N+1 samples. Fixed by inspection.
8. **The proposed strict cumulative clock check falsely rejected valid full PICO.** Binary64 `time += .002` differs from `steps*.002` by more than1e-10 starting at step60370, time120.74s; full PICO has65300 steps. This affected the new actual guard and independent oracle. Revised runner/oracle independently accumulate expected time from the initial time, never resynchronize to observed post-step time, and keep the same1e-10 comparison tolerance. Difference from ideal step-count time is exposed separately as roundoff. The saved pure regression passes all65300 normal steps, including nonzero initial time. Reset, missing and extra steps are detected immediately; added1e-12 per-step drift is detected at step100.

## Logic reviewed without further findings

- Invalid candidate lanes are latched at the first bad2ms sample and never rejoin the valid set. Nonfinite controls/costs become invalid. Feasible lane selection excludes NaN/Inf totals.
- `_ilqr_feasible` independently rerolls cached seeds; no finite feasible incumbent produces `NoFeasiblePlan`. Derivatives are not used to turn an invalid seed into a purported recovery.
- Only a strictly lower finite candidate cost replaces the incumbent. Candidate K is copied into the returned controller only with an accepted rollout. Rejected sweeps, all-Inf/NaN line searches, invalid derivatives and worse finite candidates retain the last accepted targets/states/K. Initial feasible open-loop seed has zero K.
- Hard line rollouts inspect every2ms integration result, with one derived-field forward after each20ms hold to preserve existing dynamics semantics. The planner's warmstart-reset model remains its declared prediction model; it is not claimed to be a complete copy of actual plant state.
- Imminent actual controls are separately checked using a complete private MjData copy, with no reset or forward before stepping. Actual state/history update occurs only after this guard passes. The oracle independently uses its own full copy and verifies original integration state and warnings unchanged in `finally`.
- `NoFeasiblePlan` and imminent-control rejection stop before the rejected target is applied. Rejection is explicitly not recovery or success. Actual partial control slots retain their substep counts; final full-source/probe flags remain false on failure. A final trace/report still contains the executed prefix, including zero-control rejection.
- Oracle records initial plus all post-step q/dq/time/warnings and N×23 command/actual-force arrays. It retains first failure when asked to continue a diagnostic. Its updated clock accumulator does not change physical inputs, q/dq, forces or source timing.

## Evidence

- `predicate_checks.json` and `predicate_model_snapshot.py`: independently executed pure-array checks, exact loaded source bytes bound.
- `reporting_checks.json` and `reporting_runner_snapshot.py`: JSON normalization and empty-shape export operations only, loaded source bytes bound.
- `clock_accumulation_regression.json`: normal65300-tick recurrence plus reset/missing/extra/drift witnesses, no simulator.
- `reviewed_sources/`: final read-only source snapshots, including the root-owned oracle and test module. Tests were inspected but not physically rerun by this review.

Final reviewed source hashes: model `61d1c168ba5db189fc1e882bfb1e2c9f7773584353b8e2251713ae0d0dd7dc9e`; core `1efc9cb21e446d8e41e3d3acebe05d12b951e8a0002b264ddf0a6b723e32f45a`; tracker `7695d03787de93dc3d4b1fb2001d6b6b0dcdb9016af9c96734e8f7d94a5e5dde`; runner `373b792710f0427cc56fe09b5f32f3606d837d9eaee345090312f634d06c129e`; oracle `0027210cda5a44255debecd7151ecd454e2281a5ff4e3137641d102701fb1330`.
