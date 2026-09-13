Fixed ankle-roll torque experiment improved native23 joint-limit behavior in all six tested complete replays. Full-body tracking and hardware remain unqualified.

One fixed law was tested: an inward spring within 0.05 rad of either ankle-roll limit (150 Nm/rad), plus damping only against outward movement (2 Nm per rad/s). Add this torque to BFM PD torque, then apply native effort caps, including 35 Nm ankles. All other added joint torques are zero. Original references, BFM goals/action history, and native model remain unchanged. No root forces or simulator state writes occur after initialization.

| Clip / direct arms | Baseline range excess | Fixed-law excess | Matched-source leg RMSE, before -> after | Root p95, before -> after |
|---|---:|---:|---|---|
| walk002 / no | 0 | 0 | 0.1908 -> 0.1908 rad | 0.5175 -> 0.5175 m |
| walk008 / no | 0.010048 rad | 0 | 0.1531 -> 0.1534 rad | 0.5536 -> 0.5520 m |
| PICO / no | 0.004756 rad | 0 | 0.1678 -> 0.1729 rad | 0.3297 -> 0.2968 m |
| walk002 / yes | 0 | 0 | 0.2015 -> 0.2015 rad | 0.5319 -> 0.5319 m |
| walk008 / yes | 0.009134 rad | 0 | 0.2868 -> 0.2880 rad | 0.8709 -> 0.8452 m |
| PICO / yes | 0.006052 rad | 0 | 0.1619 -> 0.1603 rad | 0.3762 -> 0.3434 m |

All six fixed-law candidates completed their full lifecycle without falls and with zero measured joint-range excess. The walk008 baseline without direct arms ended at 472/1114 controls. Its table comparison uses only the common 122 source-motion controls; the candidate completed all 364 source controls and all 1114 lifecycle controls. Other comparisons use the entire source phase. Full walk008 root p95 remains about 0.85-0.89 m, so accurate full-body teleoperation is still unqualified.

PICO without direct arms: root p95 improved about 10%, leg RMSE rose about 3%, and relative-foot p95 changed from 0.207/0.190 m to 0.205/0.199 m. With direct arms: leg RMSE and root p95 improved; relative-foot p95 changed from 0.191/0.198 m to 0.196/0.184 m. Easy walk002 never activated the law and stayed exactly unchanged with either arm mode.

All six comparisons were bit-exact before the first active barrier command. The zero-law standalone runner also exactly matched the root referee on all 13 non-timing arrays for full walk002. An incompatible old walk002_arms_v2 baseline was detected and retained in comparison_unmatched_arms_baseline_v1.json; a fresh zero-law direct-arm baseline replaced it in comparison.json.

Every fixed-law candidate was independently audited: all saved 2 ms torques reproduced qpos/qvel exactly; PD torque matched exactly; a separately coded ankle law matched within 2.7e-15 Nm; native clipping matched exactly. Six focused torque-law tests passed. Each case binds request, report, trace, source snapshots, and audit hashes.

Implementation: gear_sonic/utils/g1_true23_bfmzero_ankle_barrier.py; standalone evaluator and independent audit under gear_sonic/scripts with matching names. Optional --motion-override uses the root load_case_motion contract, preserves source count/rate, and records original/override hashes. Root retarget_v2 plain-run failures involve upper-body limits (PICO right shoulder roll; walk002 left elbow), outside this ankle-only correction. No combined retarget qualification is claimed.

Reproduce from repository root:

```powershell
python -m gear_sonic.scripts.evaluate_g1_true23_bfmzero_ankle_barrier --clip pico --arm-reference --output <new-directory>
python -m gear_sonic.scripts.audit_g1_true23_bfmzero_ankle_barrier <new-directory> --output <new-directory>/audit.json
```
