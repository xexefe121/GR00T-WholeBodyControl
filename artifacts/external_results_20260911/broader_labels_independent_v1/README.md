# Independent audit of 6,847 broader moving labels

**Both independent audits pass.** Every selected PICO row (5,980) and walk002 hybrid MPC-prefix row (867) matches its original qualified trajectory and independently reconstructed simulator state. One separately selected original BFM reevaluation also reproduces every saved raw actor output and derived baseline exactly. No dynamics, new labels, normalization change, or fitting occurred.

| Audit | Rows | Retained comparisons | Backward / actor calls | Result |
| --- | ---: | ---: | ---: | --- |
| [Saved arrays and formulas](saved_array_audit/report.json) | 6,847 | 158,521 | 0 / 0 | All exact |
| [Independent BFM reevaluation](baseline_inference_audit/report.json) | 6,847 | 185,909 | 6,847 / 6,847 | All exact |

The saved-array audit independently enumerates PICO controls 250–6229 and walk002 controls 250–1116, with received state frame `control + 11`. It rebuilds actual measured history from control 0, using its own history storage/update loop and the original floating-point quaternion expression. It checks the complete preceding history through each unselected terminal boundary, including unclipped normalized MPC previous actions and signed zeros.

Each selected row checks actual native23 target, native joint limits/span, all 291 integration values, qpos30/qvel29, repeated clock, warning ledgers, and previous actual torque against qualified source traces and independent root snapshot archives. Saved BFM state52, flat history300 and all named four-lag histories match bytes. The target-minus-unclipped-base residual and original 1,069 received-goal feature expression match bytes. Teacher boundary arrays contain the exact N+1 source states. Original normalization mean/std remain byte-identical.

The inference audit loads the original native per-clip reference, fixed horizon8/position1/yaw2 BFM goal and pinned actor/backward graphs. It directly calls them at each independently reconstructed actual state/history/frame, without importing producer validators or the producer collection loop. It compares sensor state, raw actor action, unclipped baseline target, and features byte for byte. Each graph's fixed call budget is enforced; no teacher or terminal target substitutes for the baseline. All input hashes are rechecked before PASS.

Nine initial audit-mechanics tests passed; the final inference version has ten passing tests, including generic inference-fault evidence preservation. The first pure audit's source and requests remain unchanged. The separate `source_inference_v2` adds staged goal/raw/base/features exception snapshots and start-to-finish request-hash validation. Its final [prelaunch review](../broader_label_inference_prelaunch_review_v1/review.json) is `7505d61f701450bb67751581af8207cc4cc10c7e9df87dcf459dc458e4b9bde1`.

The independent audit's hidden process completed with a recorded **exit code 0** at 08:06:54 UTC. All 139 launch hashes remained unchanged and wrapper/child/WSL processes are absent. [Completion verification](completion_verification.json) records this separately from the original collector's unavailable (`null`) child exit code, which stays preserved as unknown. A harmless Windows PowerShell 5.1 exit7 stub verified the corrected process-handle capture before this audit launched.

Evidence identities:

- Saved-array report: `afa84495b4069554730a2a5242511fe99e0ba31515e3bc9f372074bb3a34101c`.
- Saved-array comparison log: `47bbeafe5ac57662d59d2ef84c6c8c82af323a1ab822903a114e1b84f243b013`.
- BFM inference report: `d090f7d11acd92da6e4f922ceba04c546926d3ec680b5730033d78d1b708a717`.
- BFM inference comparison log: `4b2c13f8eec63c0f12abf6533fb4e65a5dfdef3b83ff228117ded6f78c8d753a`.
- Final audit source: `07f3620886639db787dea32f69c948a2278af35b53b47df3bebfc3f202255e07`.
- Final request: `a45d08b859eb8c6d42d93948911c78e8de7c82fea037ef6a93ca2d8ebf2eae78`.
- PICO labels: `28097c143fa1c8ac86c660a8945c83a59cf58bb9586778e5041fbeb98245a2df`.
- walk002 labels: `f19c08d17e2e801e169bdd8ad4018ae56efe7c2e23dc7a7e330921fff0379207`.

These results verify this offline dataset's recorded provenance and feature/baseline authenticity. They do not establish learned-controller performance, live teleoperation, real-time control, or hardware readiness. walk008 remains excluded.
