# One nominal fast-controller pilot — rejected

The range-complete BFM-conditioned student failed during initial entry: right knee crossed its native lower bound by 0.002097812845 rad after 24 controls / 240 physics steps / 0.48 seconds. No source motion reached. The 1569-control lifecycle and separate 250-control hold were not completed. No DAgger query, second fit, or additional rollout was launched.

The fixed 1069→256→256→23 linear head trained for 1000 steps (seed 773, CPU one thread) in 9.98 seconds. Applied-target training RMSE was 0.08973 rad in initial entry, 0.10213 in acquisition, 0.13823 during source, and 0.08279 during return. These are training-state metrics, not generalization or behavioral success. Per-joint errors remain in fit/report.json. ONNX versus Torch maximum output difference was 9.69e-7 rad.

Before fitting, the zero head produced exactly zero residual on all 1269 demonstration states and exactly matched the same frozen ONNX BFM prior for 100 physical controls / 1000 steps, including history and targets. zero_parity/completed_binding.json binds the completed check, graph, and frozen source receipt. This does not claim historical PyTorch-versus-ONNX trajectory equivalence.

The learned rollout used original native dynamics at 500 Hz, bounded native PD effort, and 50 Hz policy updates. Every actual substep was recorded. Warning counts and accumulated clock errors stayed zero; maximum speed ratio was 0.615725 and maximum effort ratio 1.0. Targets stayed within native bounds, but the right knee's actual state violated its lower bound. The failed 24th control completed all ten physical substeps; physical completion count does not mean it passed.

Policy timing during this short failed rollout was 9.02 / 12.96 / 50.37 ms (median / p95 / maximum), with one of 24 calls above 20 ms. This is not a qualified real-time path or a quiet-machine benchmark.

All precontrol integration states, BFM histories, previous actions, and final failure state remain in nominal/trace.npz. Qualified teacher labels used actual applied-MPC target normalization; student recurrent history used the disclosed combined preclip BFM-plus-residual action. Original full-body goal lookahead was retained: 0.74-second feature knots, up to 0.76-second raw reference support. No frame ID, time, or clip ID enters the network.

This single fit removed the old residual cap and used a qualified full walk003 demonstration, but still failed closed loop. Further oracle correction requires a separately reviewed actual-state query and a qualified full remaining continuation, return, and standing hold. A short feasible MPC horizon alone is not an expert-label qualification.
