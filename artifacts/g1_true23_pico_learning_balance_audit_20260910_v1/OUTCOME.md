# Training coverage audit: no missing-phase explanation

`analyze_v2.py` exited successfully. The first script failed at import; its
unchanged source and setup failure remain beside this result. No new dynamics
or training ran in this audit.

Both recorded 1,024,000-transition runs were joined to their exact reference
indices and checked against their existing arithmetic/hash audits. Parent500
visited 5,777/5,780 PICO source anchors; foot1000 visited all 5,780. Around the
six queried failure-region states, foot1000 has 3,883–6,754 actual transitions
per +/-25-frame neighborhood. Missing clips/phase exposure is not supported.

Across nonterminal foot1000 transitions, normalized foot-cost median is 6.35
(roughly 12.6 cm RMS); 37% of costs exceed 9. Its original bounded negative
term contributes only about -0.02/control, but the separate foot bonus already
contributes about 0.97/control around control1830. Adding that bonus did not
produce full-motion acceptance. Marginal feature overlap is not joint-state
coverage; reward allocation is not a PPO/physics gradient or proof of conflict.

No new reward/reset sweep is justified by these measurements. Earlier
failure-weighted starts, full-decoder adaptation, native124 substitution and
waist compensation were also inspected and already failed. User now explicitly
requests a changed learner and a twelve-hour target. Next isolated trial uses
a compact native23 reference-conditioned tracker, not another small frozen
decoder modification. Source inputs, physics and motion acceptance remain fixed.

Full-body teleoperation is not deployment-ready. No robot operation authorized.
