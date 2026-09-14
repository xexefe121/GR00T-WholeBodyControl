# Bounded shoulder-recovery experiment — completed

**Earlier intervention can avert the immediate failure, but none of the three tested interventions demonstrated sustained recovery. No candidate promoted.** Best fresh closed-loop trial lasted 79.72 seconds versus 63.38 seconds without intervention, then strict rejection returned at the same right-shoulder lower boundary. Tracking failed; terminal standing was not reached. Independent real-time performance was not evaluated.

## Verified reconstruction

`strict_newer_v1` replay matches all **31,696 recorded 2 ms physics steps**, with zero position, velocity, and commanded-torque differences. Full `mjSTATE_INTEGRATION` snapshots were saved at controls 3169, 3168, 3167, 3164 and 3159. Serialization continuation matched exactly at all five boundaries, including solver warm-start state. Environment: MuJoCo 3.2.3, Linux x86_64, Python 3.11.15. These are applied-target replay claims, not reconstructed controller-history claims.

The new uninterrupted controller run separately verified restored history at controls 3159–3169: ONNX features, next targets, all 100 physical steps and full integration states matched, including the same final rejection. Its packet admission times are newly defined logical times; it is not presented as an exact recreation of the earlier asynchronous run.

## Target-by-delay result

| Boundary before rejection | Sampled targets | Passed every delay, 0–12 ms | Also passed native fixed-candidate preview |
|---|---:|---:|---:|
| Rejection observation | 318 | 0 | 0 |
| 20 ms earlier | 317 | 314 | 314 |
| 40 ms earlier | 325 | 325 | 325 |
| 100 ms earlier | 307 | 307 | 307 |
| 200 ms earlier | 319 | 319 | 319 |

At the rejection observation, passing counts for delays **0, 2, 4, 6, 8, 10, 12 ms** were **286, 283, 279, 272, 259, 228, 0**.

The unchanged 12 ms prefix stayed inside the original reserve, with **0.00043946 rad** reserve remaining. Failure occurred after the replacement could act. The best sampled 12 ms result first breached at 14 ms and reached a minimum physical margin of **−0.00323435 rad** over the 32 ms horizon. Its target was −0.58663008 rad. Per-step PD effort remained capped at the configured limits.

The 257-point initial grid spans the legal −2.2515 to +1.5882 rad range, with approximately 0.01499883 rad spacing, plus measured/requested/previous/guard/saturation seeds. Two refinement rounds use approximately 0.00187485 and 0.00023436 rad spacing. Counts differ by boundary because clipped or duplicate seeds are removed. Total: **1,586 boundary-specific target samples and 11,102 target-delay trials**. No sampled candidate passing is not proof that every possible command fails.

All other 22 actor-requested targets stayed fixed during each short search, while the full coupled robot was simulated. Position, reserve, speed, applied-effort, fall, nonfinite, quaternion and engine checks remained active. Requested and saturated torque traces are stored separately. Native cold-state fixed-candidate preview results are distinct from restored warm-state physics results; the original runtime guard's verdict is also retained.

## Actual controller continuation

Three distinct survivors were selected from control 3168, 20 ms before rejection. Each passed all seven delays again on the fresh controller run's snapshot. Only that one shoulder request was intervened upon. Future targets came from the actual controller responding to the changed state, with strict rejection active.

| Trial | Shoulder target (rad) | Motion stopped at | Gain over fresh baseline | Terminal reason |
|---|---:|---:|---:|---|
| Fresh baseline | unchanged | 63.38 s | — | Strict preview rejection, control 3169 |
| Best short-horizon margin | −1.24681287 | 63.50 s | 0.12 s | Same shoulder lower boundary, control 3175 |
| Positive-effort saturation seed | −0.54304894 | 79.72 s | 16.34 s | Same shoulder lower boundary, control 3986 |
| Largest legal surviving target | +1.5882 | 63.54 s | 0.16 s | Same shoulder lower boundary, control 3177 |

All three: short preview passed; full physical completion **false**; full-body tracking **false**; terminal 30-second standing **not reached**; timing **not evaluated**. No executed physical-limit violation was detected before their strict stops. The longest branch consumed 3,636 of 5,780 source-motion controls. Its partial-run root p95 was 0.364 m, foot p95 values 0.304/0.379 m and leg RMSE 0.341 rad; these exceed the original tracking limits. Different duration prefixes must not be treated as a matched tracking comparison.

Terminal diagnostics were recovered by repeating the same three fixed branches, with every physical step checked exactly against their saved traces: 70, 8,180 and 90 steps after intervention. This did not add candidates. Full controller restoration was verified before any intervention; the supplemental repeats retained the final rejected commands and guard evidence.

These offline runs pause simulated time while computing. They do not establish 50 Hz controller deadlines with independent 500 Hz physics. Timestamp-selected intervention is a counterfactual, not a deployable trigger.

## Engineering decision

Stop this bounded study. The result supports investigating a **state-based, delay-aware braking trigger that acts earlier and remains effective over subsequent commands**. It does not justify promoting a single sampled target, increasing gains, weakening strict rejection, or claiming native23 SONIC/full-body teleop readiness. The experiment does not establish that any proposed sustained braking rule will work; that would be a separate implementation and complete-motion test. No new training recipe or hardware run was started.

## Deliverables

- [Machine-readable result](report.json)
- [Compact feasibility table](feasibility_summary.csv)
- [All 11,102 target-by-delay rows](target_by_delay_all_boundaries.csv)
- [Continuation table](continuation_summary.csv)
- [Exact replay report](replay_v1/report.json)
- [Controller restoration report](continuation_v1/restoration_verification.json)
- [Continuation reports](continuation_v1/report.json)
- [Terminal failure diagnostics](terminal_check_v1/report.json)
- [Source and asset hashes](implementation_manifest.json)

Reproduction instructions: `Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/onboard_inspection_20260912/SHOULDER_RECOVERY.md`. Focused checks: **26 passed**. Production strict runner's hash still matches `strict_newer_v1`. Original actor, model, gains and limits remain pinned.
