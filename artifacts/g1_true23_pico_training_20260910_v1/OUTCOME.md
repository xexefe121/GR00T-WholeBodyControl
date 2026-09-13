# Existing-PICO adaptation: measured progress, deployment rejected

The fixed experiment is complete. Goal remains ACTIVE. Neither checkpoint is
deployment-ready. No robot, DDS, controller-mode change or hardware export ran.
No further training run is active. Do not restart this completed experiment.

## Actual change

Earlier normal-core adaptation excluded PICO-FreeDancing. This new bank uses
the complete saved walk002/003/PICO records, retains original29 hand/head intent
and all12 physical leg joints, and excludes walk008 from optimization. It keeps
7,266 source frames and9,549 lifecycle frames bit-exact. PICO is now training
data, not an unseen generalization result. The body reference is optical-derived
from paired PICO/OptiTrack capture, not a raw-headset-only reproduction.

After the independent two-update wiring smoke, one fresh run completed500
continuous PPO updates,1,024,000 transitions and4,000 optimizer steps in1333.83s.
No actor, critic, optimizer or simulation reset occurred between checkpoints
other than normal environment episode resets. All seven LoRA factors, Root9,
exploration and critic changed; every frozen source weight remained exact.

Final checkpoint: `train500_v1/checkpoints/normal_lora_model_500.pt`.
SHA256: `047b7387f8268d2ff8032c8b74764de6a3f0321f4245814c0f5a909e805dc065`.

## Complete fixed requests

| Request |100 completed controls|500 completed controls|Requested|500 full-body tracking|
|---|---:|---:|---:|---|
|walk002|1417|1417|1417|FAIL|
|walk003|1105|1569|1569|FAIL|
|walk008, excluded from optimizer|1114|1114|1114|FAIL|
|PICO-FreeDancing|1874|3817|6530|FAIL|

PICO source completion improves30.48→69.34 of115.60s. Untouched SONIC reached
50.60s. The500 checkpoint still stops at the unchanged next-step joint-range
guard. Over the baseline's entire50.60s source prefix, root p95 improves
5.239931→0.359780m and arm RMSE0.283277→0.266280rad, while leg RMSE worsens
0.163436→0.169004rad. Relative feet remain13.98/16.77cm. Across its entire
69.34s source prefix, root p95 is1.053879m; do not quote the shorter-prefix
0.359780m as the complete candidate result.

Over the identical30.48s shared by100 and500, root p95 improves
0.678713→0.355009m but legs worsen0.143495→0.162196rad and both relative feet
worsen. The improved duration and root/arm tracking are genuine partial gains,
not full-body acceptance. Walking remains poor;002 root drift worsens and
500-checkpoint relative-foot p95 errors are approximately16–22cm across walks.

All executed500 prefixes have hard-range excess0, effort ratio<=1 and maximum
velocity ratio0.615654. A guard rejection before completing PICO is NOT a full
physical-bounds pass. Standing-entry/return qualification, source fidelity,
contact/slip, timing, fault/recovery and physical handback remain separate gates.

## Independent verification

- `train500_v1/arithmetic_audit.json`: all1,024,000 recorded rewards checked;
  4,000 optimizer steps; all trainable groups changed; frozen base bit-exact.
- `eval100_v1/independent_audit.json`:55,100 fresh2ms saved-torque steps,
  5,510 landmark vectors exact;384 independent policy reinferences exact.
- `eval500_v1/independent_audit.json`:79,170 fresh2ms saved-torque steps,
  7,917 landmark vectors exact;384 independent policy reinferences exact.
- `eval100_v1/equal_budget_comparison.json` and `milestone_comparison.json`
  compare identical source intervals and initial physical states.

Training diagnostics do not show a clean tracking-learning trend: first/last100
updates' mean pre-action feet-world RMS is0.133619/0.141944m. Median completed
reset intervals grow110→123 controls, not complete-motion mastery. These are
nonstationary on-policy populations, not a controlled fixed-state comparison.
Sampled telemetry cannot establish complete reference exposure. Full replays,
not console reward or reset-interval length, decide readiness.

## What this establishes for the next goal step

Dataset ownership and continuous training now have tested implementations.
Do not repeat another fresh100-update campaign, widen the joint guard, present
only the improved PICO prefix, or replace SONIC with the rejected native124.
Preserve500 as a research checkpoint; its leg/walking regressions forbid
promotion. The next change must improve measured leg/foot tracking and complete
PICO, while retaining root/arm gains. Any continuation needs an explicit new
experiment and validated actor/critic/optimizer preservation: the current normal
runner intentionally supports fresh runs only. Do not claim exact simulation/RNG
resume from these checkpoints; those states are not saved.

Source-method cross-check: [SONIC-Transfer's draft](https://sonic-agibot-x2.github.io/sonic-transfer/static/pdfs/paper.pdf)
uses closely matched skeletons, original tracking rewards and substantially
larger training, and explicitly separates morphology and dynamics limits.
[Any2Any](https://arxiv.org/html/2605.23733v3) uses joint scattering/padding plus
geometry/coupling corrections, followed by dynamics adaptation. Naming either
method is not evidence that this true23 controller passes. Our shared G1 leg
axes already align; new machinery must be justified by measured errors.

Normal reference admission retains920ms of received input. Even a future motion
pass would not by itself qualify low-latency live PICO, a hardware state estimator,
or Unitree's return-to-normal-standing handback.

## Visible replay and final checks

`eval500_v1/pico/FAILED_SIM_pico_69p34_of_115p60_seconds.mp4` visualizes every
saved boundary, including7s standing/entry, without running another policy or
integrating new dynamics. ffprobe verifies3818 frames,50fps,960x720,H.264,
76.36s encoded duration (76.34s measured boundary-to-boundary duration).
Video hash9b6246b81c05837e87d5786a67a0b49edb5f460e46994eb9331066734c4afec3.
Frames15s/75s visually inspected; video requested in the current Codex panel.
This is the failed simulation attempt, not a complete successful PICO playback.

At the final saved boundary, right ankle roll is0.259895739rad against upper
limit0.2618rad and moving outward at3.870868641rad/s; only0.001904261rad remains.
Root height is0.436701436m. These measured state values identify the immediate
limit risk; they do not establish that changing one joint alone repairs balance.
Do not remove the guard. Hardware damping causation is not diagnosed by this.

Final scoped run32115 EXIT0:16 tests pass in35.26s across new PICO training and
native124 comparison modules. Scoped Ruff E/F and document diff checks pass.
All training, evaluation, audit, rendering and test jobs are terminal.
