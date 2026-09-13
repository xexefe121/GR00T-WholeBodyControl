# Native target actuation

Status: implemented and checked; the bounded native dynamics pilot finished.
Both scheduled tests complete zero recordings. No model promotion or extension.

Recent factory-based training used different PD gains from the successful
native expert and trimmed0.06rad from both ends of every target interval. The
expert's legal commands enter that excluded range2,042 times in walk003,
2,156 in walk002 and11,343 in the Pico training labels. All these commands stay
inside the original native limits. At the ankle-roll joints, the extra margin
removes about23% of the legal target range.

That proves an action-space restriction. It does not prove the restriction was
the cause of the learned controllers' falls. A saved-target replay attempted
to compare actuator laws, but its original-PD open-loop baseline diverged and
fell at14.93s; that comparison was stopped and is not used as causal evidence.
It did not run a reusable feedback controller or invalidate the recorded
expert's closed-loop result.

## Implemented behavior

`NativeTargetActor` converts the retained factory/Pico prior to an equal
instantaneous native torque, then learns all23 corrections directly in native
target units. Every original legal target remains representable. Foot-phase
and velocity commands remain trainable. The actor keeps1582 causal inputs and
23 outputs; no future data or clip identity was added.

`NativeTargetEnv` and `NativeTargetController` use the original benchmark PD
gains and effort caps. Their received history contains actual applied native
targets. Only the frozen factory balance policy's action memory is converted
back to equivalent factory targets at the measured state. Expert reset labels
retain their original native targets, without margin truncation.

The extra experimental joint-limit brake is disabled for this explicit mode,
restoring the original expert actuator law. Original joint-state bounds,
velocity/effort limits, fall checks, tracking requirements and standing criteria
remain unchanged. Existing modes and the pinned Pico demo keep their settings.

The explicit simulation launcher option is `-TaskCommands -NativeTargets`
with an `-Actor` exported for this mode. Add `-IndependentClock` to run the
500Hz native plant and50Hz controller on separate clocks. This mode selects
the same focused bank used for training, passes the original native gains to
the plant, and disables only the additional experimental limit brake through
the existing ABI. Old controller modes retain their settings.

The plant still owns all actual applied-command history. The controller
converts each historical native target to equivalent factory memory at that
row's measured state; it converts the newest target at the current state.
Received full-body history retains native targets. No proposed target is
committed by the independent worker.

## Verified

- All23 outputs reach both original target bounds.
- Native PD torque matches the original contract exactly.
- Training/runtime observations agree within1.3e-7 and outputs within2.7e-7
  over64 physical controls, including changed actual commands and delay.
- Torch/ONNX output error is below6.3e-7.
- With512 native worlds, measured control-state throughput was5,112/s on CPU
  and3,518/s with the CUDA policy path. This includes the interface replay;
optimizer throughput is measured separately in training.

Independent-history import agrees with the unpaced controller within5.96e-8
in features and2.39e-7rad in targets across64 physical controls, including
changed applied commands. Report: `native_target_history_clock_v1/report.json`.
A6-second independent-clock plumbing replay completed3,000 physical steps
with exact original-PD torques and no physical failure or control deadline
misses. Training ran concurrently: two physics finishes exceeded2ms lateness
(maximum2.503ms). This establishes actuator wiring, not timing or full-motion
qualification. Reports: `native_target_clock_plumbing_v1/report.json` and
`actuator_check.json`. A dedicated timing run is required after a candidate
earns a complete-motion test.

Reports: `native_target_interface_cpu_v2/report.json` and
`native_target_interface_gpu_v1/report.json` beneath the firmware artifact
directory. The first CPU check exposed precision cancellation in a very large
clipped proposal; the new actor's straight-through expression was corrected
to preserve the exact clipped forward value, then both checks passed.

## Untrained complete-motion baseline

The new interface with unchanged learned weights fails all four recordings:
walk00212.034s, walk00312.238s, Pico23.704s, held-out walk0088.342s. Each stops
on the original joint-state bound. Thus this interface change alone is not a
working controller. Results are in `native_target_baseline_v1/report.json`.

## Bounded native training

Finished run: `onboard_factory_firmware_v1/native_target_ppo_v2` on the artifact drive.
Native MuJoCo3.2.3 mjbatch physics and CPU learning,512 worlds,64 steps/update.
The attempted200-update expert-command initialization regressed the native
baseline to8.944 total seconds across all four recordings. It was rejected.
The first process stopped after15 critic-only warmup updates, before actor PPO
updates. Its181.01seconds and15 updates remain counted against the pilot.

Recovery restores every original pre-fit actor parameter and retains the
expert input normalizers; the zero correction output makes that normalization
change output-identical. Critic and discriminator restarted fresh. The recovered
run received185 updates, with checkpoints85/185 corresponding to total
pilot updates100/200 and a36-minute wall cap. No repeated command
fitting or extra update allocation. Recovery details are in
`native_target_initialization_recovery_v1/report.json`. No supervised auxiliary
loss during PPO. The physical motion-prior reward and all original tracking
objectives remain. No paid compute or recurring automation.

The recovered baseline reproduces the untrained native interface's56.318 total
physical seconds across the four recordings. Training was confirmed live on
the recovered state; initial full-loop throughput is about3,500 controlled
states/second. This is training progress, not a successful controller result.

Start-state variation, received-input interruption training and50% canonical
full episodes remain. The remaining reset roles are25% expert movement,
12.5% transition/input loss and12.5% actual standing. Actor initialization uses
the retained Pico controller. Training and final evaluation now share the
original native simulator and actuator contract. Fixed complete rollouts decide
whether this removes a practical obstacle; no extension or promotion on loss.

The first scheduled fixed test (local update85, total pilot update100) completes
zero recordings: walk00212.656s, walk00312.212s, Pico23.718s and walk0088.362s.
Each ends on an original native joint-state bound; every full-body tracking
verdict fails. Total physical duration56.948s versus56.318s before PPO is not
a useful complete-motion improvement. The second fixed test ran at
local185/total200. Details: `native_target_ppo_v2/eval_00085/report.json`.

## Final fixed result

The process exited successfully after185 remaining updates and1,995.328seconds.
Including the rejected initialization run, the pilot consumed200 updates and
2,176.343seconds. There is no active training job or automatic extension.

Final physical durations: walk00212.806s, walk00314.524s, Pico41.312s,
walk0088.386s. Zero full recordings, zero30-second terminal holds, all tracking
verdicts failed. Total physical duration increased to77.028s from56.318s before
PPO, but does not earn promotion. The retained factory-gain Pico controller
still has the stronger complete-motion result.

The final native joint-bound stops are right wrist roll in walk002, right
shoulder roll in Pico, and right ankle roll in walk003/walk008. Foot tracking
also fails (Pico foot errors0.250/0.304m over the completed source prefix), so
a joint-limit guard alone would not establish full teleoperation. No bounds
or tracking criteria were relaxed. Final report:
`native_target_ppo_v2/eval_00185/report.json`.

The final Pico rollout is rendered at original speed in
`native_target_ppo_v2/eval_00185/pico/video_fast_caption_v2/pico_learned_received_unpaced.mp4`.
It contains all41.312s/415 frames of this failed run, not the requested160.6s
complete lifecycle. Captions explicitly identify a partial rollout and failed
tracking. Frame timestamps were checked; a shared camera preserves root error.
The contact sheet was visually inspected. No new physics, root alignment or
time warp was used. The earlier caption version is retained separately; use
`video_fast_caption_v2` for the clear partial-run label.
