# Frozen native23 BFM with bounded learned target residual

This is a distinct simulation-only experiment authorized during the six-hour
23DOF full-body teleoperation task. It starts from the public native23 BFM-Zero
controller that completed saved PICO and walking lifecycles. It does not extend
the failed scratch compact learner or use any SONIC/LoRA weights as this actor.

The BFM actor/backward-map tensors remain frozen. The learned actor has188
inputs, two256-unit ELU layers and23 outputs (120,110 parameters including its
Gaussian exploration parameters). Its last linear layer starts exactly zero.
PPO samples unrestricted pre-tanh residual variables; the executed target
offset is0.15*tanh(u) radians per native joint. A separate critic is learned.

The base uses the public BFM PD/action contract, measured-heading goal-velocity
feedback with position gain1/yaw gain2, and the mean of eight received goal
embeddings. Joint targets are clipped to the native hard ranges. PD is computed
every2ms and saturated at the existing full native effort limits. Geometry,
inertia, passive joint parameters, contacts and solver remain the pinned
native23 model. There are23 physical actuators and no phantom joints.

BFM state52, last action23 and history300 follow the public evaluator. History
contains four previous samples, newest-first within each alphabetic term.
History and previous action reset to zero at reset. The previous action is the
combined preclip BFM/residual action-equivalent, preserving exact zero-residual
behavior even when physical targets clip. Residual observation188 is the
existing native compact165 feature vector plus the current base target23.

The original existing-PICO bank includes walk002, walk003 and full PICO.
Previously seen development clip walk008 stays outside optimization. Training
retains25% standing starts and75% sampled source reference starts; reference
state initialization occurs only at environment resets. Training uses the
existing bounded original-task/root/foot objective plus world-quality and
foot-precision bonuses. Actual joint-range/velocity failure, checked during
physics and after the control step, matches the CPU referee: range excess above
0.01rad or speed above the full native velocity limits terminates the episode.
The measured joint-limit reward remains. Source targets and physical thresholds
are never changed to make an evaluation pass.

Validation before the main run:

- Batched state/history match CPU within3e-8, feedback goal within2.9e-6,
  and base action within1.4e-5; CUDA action error is below8e-6.
- Zero residual reproduces the base target exactly.
-16 actual GPU input states reconstructed independently on CPU match188
  features within5.96e-7; functional CPU actor matches within1.4e-8.
-64-env two-update smoke and256-env two-update continuation passed all reward,
  normalization, finite-gradient/weight and source/physics checks. Warm256-env
  update takes about3.7s. These small runs are not full-body success evidence.

The main run resumes actor, critic, optimizer and normalization from smoke
update4; environments and histories reset explicitly. It runs for at most three
hours, saves every200 optimizer updates and a final checkpoint, and retains
metrics and unsuccessful replay evidence. Root task owns complete CPU replay,
original29 task metrics, stream lifecycle tests and the final behavior decision.
No robot transport, motor command, hardware arming or physical test is part of
this experiment.
