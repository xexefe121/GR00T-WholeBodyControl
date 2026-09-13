# Bounded private restoration with zero rollout feedback

One control3805 restoration solve used the same model, initial full physical state, shifted targets, physical-margin merit, backward algebra, nine alpha values and maximum10 iterations as the failed guided solve. Only the private forward rollout replaced the backward sweep's K with zero. The feedforward direction k was unchanged. This change did not modify any physical controller, main tracking objective, native limit or original source timing.

Merit decreased4.292815755335639→0.00025182496167593587. Every iteration accepted an improvement. Of90 generated candidates,64 passed both the strict every2ms nominal rollout and the independent native manual-PD continuation from the complete actual MjData state. No actual control was executed.

The ordinary final returned target sequence is exactly `feasible_10_8.npz`, SHA256 `a8a6a705b084ffd2e4ac65620e9a97ada6c145b8b12d73b6d3df53a3fe8b23f6`. This is not a hindsight-selected alternative. All300 native substeps pass: joint excess0, maximum speed ratio0.82127153, effort ratio1, maximum tilt0.37034504rad, minimum root height0.29990686m, zero warnings and zero expected-clock error. The unchanged main tracking cost is1137.26082749.

For comparison, the original shifted seed was independently continued through its entire unsafe horizon. It violates only right ankle roll upper range, with maximum excess0.0028921013rad and final excess0.0027966882rad. The initial crossing at570ms is much smaller. All other native predicate components pass, including minimum height0.27851381m and zero warnings. Full raw trace is `initial_full_horizon.npz`.

`all_generated.npz` retains the original seed, all90 generated target sequences, all backward feedforward directions and feedback matrices, final targets and nominal states. The report binds the input and executed source hashes; source snapshots are present. The saved backward K is diagnostic provenance only: it was suppressed during this experiment's forward rollouts and is not an execution controller.

This is a certified0.6-second seed, not a complete recovery policy, full-source result or live teleoperation controller.
