# One supervised-fit continuation — rejected

The ordinary final 20,000-step student failed its single native physical attempt at 0.312 seconds: right hip-roll velocity reached -21.25169 rad/s against the native 20 rad/s limit. It completed 15 control intervals plus six substeps of the next interval (156 physics steps). No source motion was reached; the 1569-control lifecycle and separate continuous 250-control hold were not completed. No DAgger query or extra fit was run.

All 15 previous v2 source files, including runtime and evaluator, were copied byte for byte into source_snapshot_v3. Only continue_linear_fit.py was added. Same 1269 labels, features, normalization, network, joint-span output, phase sampling, seed 773, AdamW settings, and CPU one-thread execution were retained. The original failed pilot remains untouched.

Before step 1001, the original first 1000 updates were regenerated. Every model tensor, normalization/span, all ten logged losses, and all saved teacher prediction arrays matched exactly. Reconstructed optimizer moments and Torch/NumPy random states were saved, then restored for continuation. The original optimizer was not saved, so this is deterministic reconstruction evidence, not a comparison against nonexistent original optimizer bytes. fit/reconstruction_parity.json contains the receipt.

The single continuation ran to the declared maximum of 20,000 total steps, taking 179.37 additional seconds. Full/per-joint teacher metrics were recorded every 1000 steps. The early-stop conditions—full1269 applied-target RMSE <=0.01 rad AND each first24 joint's p95 absolute target error <=0.03 rad—never passed. Final values were 0.02684 and 0.15703 rad respectively. No intermediate physical tests or best-checkpoint selection occurred; the ordinary final actor was used. ONNX/Torch output difference was at most 1.82e-6 rad.

| Measure | Original 1000 steps | Final 20,000 steps |
|---|---:|---:|
| Initial input target RMSE | 0.10864 rad | 0.02409 rad |
| Source teacher-input target RMSE | 0.13823 rad | 0.02744 rad |
| Initial right-knee target error | -0.21701 rad | +0.02614 rad |
| Initial right ankle-pitch target error | -0.15803 rad | +0.03590 rad |
| Failed physical time | 0.480 s | 0.312 s |

At control 0, all feature/state/history/previous-action/base-target arrays still match the teacher input exactly. After one 20 ms interval, actual q differs from the teacher by up to 0.00782 rad and qvel by 0.51274 rad/s. Actual-state target RMSE is then 0.09801 rad, compared with 0.02306 rad on the saved corresponding teacher input. By control 5, these target errors are 0.75063 and 0.02079 rad. Better fitting on the nominal trajectory did not prevent rapid error growth away from it.

At failure, the right hip-roll BFM base target was +0.14727 rad and the learned residual -2.29098 rad, producing the applied target -2.14370 rad. Actual speed, rather than commanded target range, triggered the stop. Actual range excess remained zero, actual effort ratio never exceeded 1.0, every engine warning counter was zero, and recorded time matched the independently accumulated 2 ms clock exactly.

The short failed rollout's policy median/p95/maximum timing was 9.46/21.58/51.55 ms, with one of 16 calls above 20 ms. This is not a qualified real-time or quiet-machine benchmark. Every precontrol full integration state, history, and previous action, plus the final partial-control failure state, remains in nominal/trace.npz. Physical continuation and source qualification remain unachieved.
