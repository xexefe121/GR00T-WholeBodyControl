# Native23 full-body simulation status

Updated 2026-09-11 16:58 UTC. Fast online full-body teleoperation is not qualified. Continue simulation work; real Pico/DDS/robot testing remains outside this stage.

The new full58-feedback student completed exactly10000 updates from step55000 to65000. Training error improved: nominal0.60%, finite-state response40.56%, physical response1.17%. Independent saved-fit audit passed57180checks; same-weight FP64 export passed the original1e-5rad preclamp gate at2.9388367e-8rad. WSL batch1 deployment witness passed. These are training/export results, not stability proof.

Its ONE actual canonical run failed at control296/substep9, time5.937999999999568s. Right knee speed reached21.148101027651755rad/s against native20rad/s limit. It completed2969physics steps and issued297controls, including47learned commands after250BFM startup controls. Source motion and continuous standing hold were not reached. Joint range, effort, warning and clock gates were unchanged. No root force, reference edits, cropping or timing changes.

Root independently replayed all2969steps through the original native PD/strict oracle. Positions, velocities, command torque, actuator torque, time and warning fields were byte-exact. Full-source intent and quiet-standing requirements remain false. Raw evaluator exit0 means diagnostic completed; wrapper exit2 reflects the physical failure. Both actual processes are absent and all5233launch pins remained exact. No repeat of this rollout is selected.

Evidence under E:/codex-artifacts/sonic23_teleop_resume_20260911:

- direct_target_full_state_evaluation_v1/nominal/trace.npz:8eade82c93fa9c3104fcbb5807bb598005e3e19c4c6865be548906ea4adc80d2
- direct_target_full_state_evaluation_v1/evaluation_completion_verification.json:3e8c5b90d0ed53d4fe627c889d01a8806cf39475cb5027f80d6dcc733d388c00
- direct_target_full_state_independent_physics_v1/report.json:357168b1224acec1725705f59b9d93144ecebbdf8f4b636ebe92cfebb45e2973
- direct_target_full_state_fit_independent_v1/results_v1/report.json:2773ae80c0e5a85975b15eb436cdb709c1243d475cc91f9385cbafb41e757367

Saved finite-response analysis explains why lower training loss is insufficient evidence: final54-cell error remains1.138× a zero-response reference and98% of remaining paired error is antisymmetric. A projected previous-target hold is22× worse than the trained nominal output; it is not a useful controller substitute. These results do not identify whether capacity, loss interference or missing causal history is the main limitation. Completed runtime/history/fixed-map audit passed6772 checks; actual action feedback and history were correct, first feature departure251, first clipping257.

The next selected experiment compares two matched3000-update conditions from the same65000 weights: true incoming prior23/history300 versus normalized context zero. Both use1323->256->256->23, zero new weight columns, identical data/schedules/coefficient and lower cosine learning rate1e-5->1e-6. Source and saved-context review passed; all9899 chronological nominal transitions and3054 physical contexts matched. Physical context deliberately uses inverse actually applied target, which differs from old raw prior in1463 rows. Training launch preparation is underway. Both fixed endpoints will be retained; no controller endpoint is selected yet. Additive deployment source passed76 synthetic tests.

One recorded-expert process-clock benchmark ended with Linux timeout124. All3718 input pins remained exact and all observed processes are absent. Two entry model buffers matched the expected native model; no final trace, epoch receipt or final counters survived. Actual native step count is unknown. Saved file times leave only2.344 seconds after worker readiness inside the total timeout, while full fixed replay needs36.38 seconds. This attempt does not establish timing or physical qualification. A separate setup/execution/preservation watchdog design is proposed; no rerun selected and no2ms/20ms/physical limits changed.

Previously qualified slow expert results remain available: full recorded PICO6530+250controls, walk0021417+250, and query250walk0031569+250, each with independent native replay. PICO full-motion video and native replay launcher remain usable. They demonstrate offline23DOF feasibility, not a fast controller operating on live Pico input.

Acceptance remains the same fast controller across full PICO/walk002/walk003/heldoutwalk008, original tracking/source coverage, every2ms physical limits, measured quiet final3s and continuous5s hold. Independent500Hz plant/50Hz policy timing, received-input-only behavior, input fault handling/rearm and perturbation recovery remain required. Current references use prepared preview and ground-truth root; raw Pico/sensor-only behavior remains unqualified.
