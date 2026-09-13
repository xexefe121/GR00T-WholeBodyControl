# Main v1 numerical-audit stop, not completed training

Process73693 stopped after5 completed PPO updates/30720 completed-update
transitions, during the next rollout. Its physical rollout is partial; do not
call2000 completed. Only milestone0 was saved, so this run is not resumable from
update5. Metrics and sampled inputs remain unchanged.

Failure: CPU reduction of21 captured base-reward terms differed from the GPU
reward by3.0517578125e-5 at a terminal reward near-100. Returned and PPO-stored
reward errors were0. The fixed3e-5 audit tolerance was numerically inadequate.

Version4 uses a float32 summation/multiplication error bound gamma_n*sum(abs)
(n=2*21+2), float64 reconstruction, exact returned-reward matching, and a
rounding-scaled timeout addition check. It also saves an explicitly partial
learner snapshot on interruption. No reward equation, learning hyperparameter,
source, physics or physical acceptance threshold changes.

Rerun smoke first; new main is a declared fresh start, not hidden continuation.
