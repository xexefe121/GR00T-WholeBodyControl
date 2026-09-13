# Native23 full-body command training

The command-space pilot finished without improving the complete-motion result.
Its learned action could change twelve legs and five gait/velocity commands;
the upper eleven joint targets stayed fixed to received q/dq. This experiment
adds full-range learned corrections for waist and arms around that baseline.
It does not claim that additional freedom will solve balance or tracking.

There are 28 sampled latent commands: all 23 joint corrections, two foot phase
offsets, and three factory velocity corrections. Sampling deviations remain
0.03rad joint targets, approximately 0.3rad phase and 0.05/0.05/0.2 m/s/m/s/rad/s
velocity near zero. PPO likelihood and KL use those same 28 coordinates.
All original root, foot, hand, head, leg and quiet-standing objectives remain.
The model outputs the existing 23 native targets through the same 1582-feature
received-only interface. Neither clip identity nor future poses are inputs.

The native preflight exercises every joint command separately, confirms that
upper output weights receive gradients and move targets in an optimizer step,
and tests sampled first steps from actual canonical states. Those checks pass.
Zero corrections reproduce the retained baseline within 2.69e-7rad; the exported
ONNX agrees with Torch within the same bound. This is implementation validation,
not a full-motion qualification. Evidence: `full_body_commands_check_v1/report.json`.

Run: `onboard_factory_firmware_v1/full_body_command_space_ppo_v1`.
Native MuJoCo 3.2.3/mjbatch, 128 worlds, 64 rollout steps, CPU policy training;
75% complete actual-start episodes, 12.5% transitions/input loss and 12.5%
standing states; 2–4ms training command delay. walk008 remains held out.
Twenty value-only warmup updates; 200 total updates or 45 minutes maximum.
Fixed checkpoints 100 and 200 run all four complete motions plus 30s standing.
No supervised bootstrap, paid compute, hardware commands or automatic extension.
The pinned Pico demo and factory dance baseline remain unchanged.

Use the live process and `running.json` to distinguish training, evaluation and
completion. Training-episode completions alone cannot qualify a checkpoint.


## Finished result

The process exited successfully after 200 updates in 855.76s.
Both scheduled fixed checkpoints failed every recording. No promotion or
extension. The implementation works, but this training recipe did not produce
the requested controller.

| Fixed model | walk002 | walk003 | Pico | held-out walk008 | Completed with 30s standing |
|---|---:|---:|---:|---:|---:|
| Initialization | 11.006s | 15.038s | 160.600s | 10.108s | 1/4 |
| 100 | 10.978s | 15.090s | 51.822s | 10.112s | 0/4 |
| 200 | 12.386s | 14.822s | 51.884s | 8.538s | 0/4 |

At update200, the three training recordings fall; held-out walk008 crosses a
native joint bound. All full-body tracking verdicts fail. Pico root p95 is
0.213m and feet p95 are 0.246/0.310m over its completed portion, but it falls
at51.884s. Those partial-trajectory metrics cannot substitute for full-duration
tracking or the final standing hold. Update100 also falls on Pico at51.822s.

No independent-clock qualification was attempted for these failed candidates.
The preserved Pico model and default simulation launcher are unchanged.
Raw evidence: `running.json`, `checkpoint_results.json`, and each recording
report inside `eval_00100` and `eval_00200`.
