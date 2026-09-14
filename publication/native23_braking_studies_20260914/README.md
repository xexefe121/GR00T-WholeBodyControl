# Native23 braking experiments — source and complete evidence

Both bounded studies are complete. Neither produced a teleop-ready controller.

| Experiment | Outcome |
|---|---|
| Shoulder recovery | Exact replay of 31,696 physics steps. No sampled target passed all delays at rejection; earlier intervention extended the fresh run from 63.38 to 79.72 seconds, then the same shoulder rejection returned. |
| Sustained state-triggered braking | Implemented TRACK/BRAKE/RELEASE and 49 delay-pair predictions. Both original-start attempts stopped after 20 ms because the new terminal settling requirement failed. Filter remains disabled. |

The sustained filter's stop was a failure of that experimental design, not evidence that the original physical constraints were violated. Neither experiment passed full motion, tracking and terminal standing. Independent timing remains unqualified.

- [Shoulder-recovery results](shoulder_recovery/RESULTS.md) and [feasibility table](shoulder_recovery/feasibility_summary.csv)
- [Sustained-filter results](sustained_braking/RESULTS.md) and [frozen configuration](sustained_braking/archived_delays_v1/preregistered.json)
- [Download complete traces, snapshots, predictions, source and native binary](https://github.com/xexefe121/GR00T-WholeBodyControl/releases/tag/native23-braking-studies-2026-09-14)
- [Shoulder reproduction instructions](../../artifacts/onboard_inspection_20260912/SHOULDER_RECOVERY.md)
- [Sustained-filter reproduction instructions](../../artifacts/onboard_inspection_20260912/SUSTAINED_BRAKING.md)

Source is committed in its normal repository locations. Readable reports and tables are included here; the release archive includes all generated evidence, including the exact executed source snapshots. Existing checkpoints and prior baseline evidence remain in the earlier full-work and preview-repair releases.

Validation: shoulder study 26 tests passed; sustained filter 32 passed and three MJLab-dependent tests skipped on Windows. All 147 native/Python predictor comparisons matched exactly. Filter/controller restoration and a focused disabled-baseline comparison matched exactly. No robot execution was performed.
