The selected component-selective clipping-feedback experiment failed during acquisition at control 314, native substep 6, simulation time 6.292 seconds. It completed 314 full controls and 3146 native steps. No source-motion controls or continuous end hold ran. The original raw-feedback 75000 candidate failed later, at 6.386 seconds; this variant does not establish a stability improvement.

The first strict failure was left_knee_joint crossing its lower native position limit: −0.09422501051735958 rad versus −0.087267 rad, excess 0.006958010517359586 rad. Velocity was−3.5269742369582104 rad/s. Applied target was−0.087267 rad, clipped from raw proposal−0.13847385786655955 rad. Maximum recorded native speed ratio was 0.48894137588095654. Position limits remained strict and unchanged.

All seven comparison gates passed. The first 2650 native steps and complete integration/state at control 265 were byte-exact to the original raw-feedback run. The first changed prior was control 265; actor lag-history first changed at 266, exactly as declared. The original 250-control BFM prefix and reused first-head witness also passed. No new witness call was made.

Saved-array verification checked every command: all unmasked float32 raw combined-action components stayed bit-exact, masked components equal the applied-target inverse, and the next actual prior equals `feedback_action`. The original `action` and explicit `raw_combined_action` logs remain unchanged proposal values. There were 38 learned commands with feedback clipping, affecting 57 joint-command components across 65 learned commands. These results isolate the selected later feedback change; earlier learner divergence remains outside this intervention.

All 1619 launch input hashes remained unchanged. The durable wrapper exited 1 at 2026-09-11 11:36:33 UTC with no wrapper exception; wrapper 26320 and child 25212 were absent after completion. Full 291 failure state, raw session outputs, histories, proposals, masks and partial native evidence remain saved. Root owns the independent native replay and any next experiment selection. No follow-on run, fit, expert query, or hardware operation was launched.

Immutable result subjects:

- Trace: 454b8c577ef64edc1128cd8e5e9d295c2936f2ba854ddd9aeba0286ff112eaa7
- Report: 3176d422b5f4accf4ef64b641513cf983426f7af0ed5f3dc44f544afc66dfc9a
- Full failure witness: 04a1799c1733c57fe2fb8cfbc83b2db1fe3dea1a62413a7eae2c6c77336fae64
- Owner saved verification: 0be866e5782e08270d5ef2f4ec62e5688d9be874ad8c9d0a4760ac90cf69f4f7
- Final prelaunch review: de9295ec1fc09aab31ea58d81d93e003b60e165864e2f089cc226639d39245dd
- Run binding: f901d23b8ece3471f4f99c1c3b3529d4d5e20d68f4f4490551dbcb3ef8ebc840
- Unchanged 75000 head: fb856003734acc0338586482968a7e31553a11826e549a8b486e0662e4934e81
