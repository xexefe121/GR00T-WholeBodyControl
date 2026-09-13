# Received-pose native23 Pico simulation result

[Watch the complete160.6-second independent-clock Pico run](<E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/received_sim_runs/20260913_044020_384_pico/video_fast/pico_factory_received_independent_clock.mp4>)

Full-body teleop remains unqualified. This artifact records a runnable simulation demo and its remaining limits.

Recorded zero-spin Pico run: 160.600s; physical completion True; continuous30s quiet hold True; control misses 0; late physics ticks 0.
Controller median/p95/max: 1.023/1.420/2.688ms.

The controller uses the native-trained checkpoint, actual applied-command history and leg target filter alpha0.9. Physics advances independently at500Hz; the controller and received-packet producer run at50Hz. The fault-capture option stops the gait only after the generated braking reference is stationary and the robot enters its quiet standing region. Neural balance feedback and the explicit-rearm latch remain active. No future motion, per-motion gains, root forces, state resets or physical-limit changes are used. Robot feedback is privileged MuJoCo floating-base state.

| Recording | Physical seconds | Completed | Quiet30s | Control misses | Late physics ticks | Full-body tracking |
|---|---:|---|---|---:|---:|---|
| pico | 160.600 | True | True | 0 | 0 | False |
| walk002 | 11.038 | False | False | 0 | 0 | False |
| walk003 | 12.586 | False | False | 0 | 0 | False |
| walk008 | 10.124 | False | False | 0 | 0 | False |

Pico source errors: root p95 0.247m; feet 0.280/0.350m; leg RMSE 0.312rad. The existing limits remain0.20m root,0.12m each foot, and0.15rad legs. Hand/head objectives also remain enforced.

| Scenario | Completed | Required quiet30s | Control misses | Late physics ticks |
|---|---|---|---:|---:|
| pico_plus003 | True | True | 0 | 0 |
| pico_minus003 | True | True | 0 | 0 |
| pico_input_loss | True | True | 0 | 1 |
| spin100_nominal | True | True | 0 | 3 |
| spin100_input_loss | True | True | 0 | 0 |

The velocity perturbations were verified from saved initial physical states. Primary nominal, perturbation and walking runs use zero-spin timing. The standing-capture option affects only latched input faults; it was disabled in those nominal runs. Input loss occurs at control2000 (40s). Capture-off input loss stayed upright but kept stepping. Capture-on with zero spin settled quietly but had one late physics tick. The100-microsecond spin trial passed input-loss standing and timing, but its nominal repeat had three late physics ticks. It is not the default; zero-spin timing is retained. All results remain preserved. A complete zero-miss run is demonstrated; repeatable timing across all scenarios remains unresolved.

Run from PowerShell:

```powershell
& 'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof\artifacts\onboard_inspection_20260912\RUN_PICO_NATIVE23_DEMO.ps1' -Render
```

No hardware publisher is launched. Both bounded training pilots finished and stopped. Earlier Orin access was read-only; the latest connection check reports this laptop Ethernet disconnected.

Evidence:

- [pico](<E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/received_sim_runs/20260913_044020_384_pico/report.json>)
- [pico_input_loss](<E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/received_sim_runs/20260913_045639_671_pico/report.json>)
- [spin100_nominal](<E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/received_sim_runs/20260913_050119_402_pico/report.json>)
- [spin100_input_loss](<E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/received_sim_runs/20260913_050425_621_pico/report.json>)
- [walk002](<E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/received_sim_runs/20260913_045407_096_walk002/report.json>)
- [walk003](<E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/received_sim_runs/20260913_045432_434_walk003/report.json>)
- [walk008](<E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/received_sim_runs/20260913_045453_493_walk008/report.json>)
- [pico_plus003](<E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/received_sim_runs/20260913_044427_039_pico/report.json>)
- [pico_minus003](<E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/received_sim_runs/20260913_044738_052_pico/report.json>)
