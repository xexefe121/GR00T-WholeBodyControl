# Compact scratch learner recovery

Recovered run train2000_v2 stopped without outcome.json, interrupted snapshot,
or a live WSL process. Its final complete metrics row is optimizer update1015,
6,236,160 transitions and4285.81s measured learner runtime. Last persisted
checkpoint is update1000 with16,000 Adam steps per optimized parameter. This is
not a completed2000-update run. A hard process/WSL stop is consistent with the
missing finalizers; the external shutdown cause is not proven.

Replayed checkpoint1000 on all complete original saved lifecycles:

| Motion | Completed/requested controls | Source seconds reached | Failure |
| --- | ---: | ---: | --- |
| PICO |604/6530 |5.08 | Physical height/tilt collapse |
| walk002 |525/1417 |3.50 | Physical height/tilt collapse |
| walk003 |481/1569 |2.62 | Physical height/tilt collapse |
| walk008 |406/1114 |1.12 | Next-control joint-range preview failure |

Checkpoint200 had failed all four atcontrol109 before source entry. Update1000
learned stable standing (about1.5cm root error throughcontrol200), then lost
tracking and balance after source transition. PICO source root p95 is0.278m;
walking source root p95 values are0.699/0.956/1.073m. None passes full-body teleop.

For PICO/walk002/walk003, the CPU range preview changed zero actions, source
target projection was only floating-point roundoff, torque saturation was zero,
and maximum motor velocity was below56% of its allowed value. Physical hard
range excess stayed zero. On walk008 the preview first altered control405,
after root error had already reached0.958m atcontrol400. Physical action guards
therefore did not cause the original tracking failure.

A deterministic GPU comparison first retained training gyro/gravity noise and
the priming phase. A second diagnostic disabled only observation noise and
aligned the initial reference frame to CPU. It matched initial features within
3.63e-8 and actions within2.39e-7. State difference stayed below1.8e-4 through
control200, then grew during unstable source motion. GPU training terminations
occurred atwalk002423, walk003458 andPICO592. This confirms a weak learned
controller in both engines, not a gross action/input/physics parameter mismatch.
The existing13 compact feature/action/reward tests passed.

Diagnostics and scripts remain beside the original evidence. eval500 was
started then stopped because it would not distinguish a concrete defect.
Original compact learner, actor, reward and reference source files were not
edited. Work moved to a distinct residual learner over a stable frozen native23
BFM base; no unchanged scratch-training extension was launched.
