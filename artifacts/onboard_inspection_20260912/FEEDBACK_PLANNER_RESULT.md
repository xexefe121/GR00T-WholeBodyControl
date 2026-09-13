# Received-only feedback planner result

**Stopped: worse complete-motion behavior and unsuitable control latency.**
The retained Pico demonstration and factory23 dance are unchanged. Neither
planner version completed walk002; no hardware command was sent.

The experimental planner searches 32 legal twelve-leg target corrections for
two generations. Every candidate advances native MuJoCo 3.2.3 for 400 ms with
the actual PD gains, limit brake, effort and speed limits. It replans every
100 ms of simulated time. Upper-body targets remain tied to received poses.
Predicted references use only owned packets and declared constant-velocity
extrapolation. There is no recorded motion suffix or motion-specific plan.

The first version retained factory balance but held the learned correction
constant during prediction. The second executes the complete retained learned
head, received history, factory gait/velocity updates, actual-command memory,
and alpha 0.9 target filter throughout each prediction.

| Version | walk002 requested | Physical duration | Failure | Control p95 / max |
|---|---:|---:|---|---:|
| Factory feedback, frozen current learned correction | 58.34 s | 8.498 s | native joint speed | 183.84 / 232.19 ms |
| Complete learned feedback | 58.34 s | 7.594 s | fall | 198.08 / 279.72 ms |

Both fail full-body tracking and end before the terminal 30-second standing
hold. The second run reaches only 30 of 667 source controls. It is not an
improvement over the retained controller. Both runs were synchronous, unpaced
behavior experiments; neither qualifies independent 500 Hz physics / 50 Hz
control. No asynchronous runtime integration or additional training followed.

The complete-feedback implementation agrees with the retained ONNX actor to
8.95e-8 across twelve sampled inputs from the four recordings. Its 400 ms
closed-loop prediction agrees with ordinary MuJoCo plus the running controller
to 2.67e-7 in the checked state. These are implementation checks, not evidence
of motion completion, tracking, or robustness. Correcting the prediction
implementation did not repair the planner's physical behavior.

Raw runs and checks:

- `E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/feedback_planner_v1/walk002`
- `E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/feedback_full_policy_planner_v1/walk002`
- `E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/feedback_full_policy_planner_v1/check.json`

The optional implementation remains available for research in
`gear_sonic/utils/g1_true23_feedback_planner.py` and
`gear_sonic/native/true23_feedback_shoot.cpp`. It is not selected by the Pico
demo launcher. The model, reference predictor, and fixed-offset search tested
here are rejected as a live controller; no automatic extension is queued.
