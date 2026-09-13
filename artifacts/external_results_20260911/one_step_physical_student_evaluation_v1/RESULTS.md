The selected ordinary-final 75000 controller failed simulation qualification during acquisition. It completed 319 controls and stopped at control 319, native substep 3, at simulation time 6.386 seconds. No source-motion controls or post-lifecycle hold ran.

The first strict failure was left_ankle_roll_joint crossing its upper native position limit: 0.26654404676883836 rad versus 0.2618 rad, excess 0.004744046768838384 rad. Velocity was +2.504358032796864 rad/s. The applied target was clipped to 0.2618 rad from a raw proposal of 0.3319277175123412 rad. Native speed ratio stayed at or below 0.7594684280266637 over the recorded run. This is an incomplete lifecycle, not a successful full-body teleoperation result.

The original 250-control BFM prefix, actual query250 inputs, and the exact first output against the separately generated WSL witness all passed. The witness used exactly one head call, no BFM calls, and no physics. The actual run retains the original raw combined-action prior/history convention. Saved proposed target, applied target, normalized applied-action diagnostics, all precontrol histories and full 291-element integration states were preserved. Applied-action diagnostics were not fed back into the controller.

The first learned-phase target clipping occurred at control264; 47 of 70 learned commands had at least one clipped target. The separate original-BFM standing prefix also has a clipped command at control4. These counts describe saved output only and do not establish the cause of failure.

The durable canonical wrapper exited 1 at 2026-09-11 11:14:35 UTC, with no wrapper exception. Wrapper3168 and child2900 were absent after completion. All 1,569 launch input hashes remained unchanged. Root independently replayed all 3,193 recorded native steps exactly; its physics report is student_physical_response_independent_physics_v1/report.json, SHA ede8ff939da69885f509785e62e6c6cc8d0ecb190b2cb8fb15eb1e5ec4394133.

Immutable result subjects:

- Head ONNX: fb856003734acc0338586482968a7e31553a11826e549a8b486e0662e4934e81
- Main trace: 38428ae27055be7c2e771c5a23056c858f004ef38f260a0b45a57662d8d0aae3
- Main report: 4a81e6c4d352116efe20673b36e7634fd8d3ac710ca3f00e4046bdd9b1b968f1
- Full failure witness: ee269de1ae6c2c3973f6f2cdaa56051121c82db257d09a95b882428a7f7b9e67
- Owner saved-evidence verification: 808ca6ad5d4db74bc0597653b2779b71f457369581f1e3d587d50194ddc90e92
- Single-head witness: fa2dad8a24f43f577ec1120a86e41d7e76f6d09805cc0e637b14907c0a56e088
- Final canonical review: 04805c6947edfd146b6314c4fe62d744d5201dcf3e3a6ea6776493750b3b7bbe

No additional fit, model query, controller candidate, or physical rollout was launched. Root and the reviewer own the next saved-state diagnosis and any subsequent selection. Hardware remains unqualified.
