# Fixed continuation finished; no deployment-qualified checkpoint

Training, both full-motion evaluations and independent replay audits are
complete. No job in this experiment remains active. No automatic extension.
The full-body PICO goal remains active, but hardware deployment is NOT READY.
No original repository, recorded source, native limit, robot mode or motor was
changed. Neither a longer prefix nor a visually upright robot is acceptance.

## Implemented and executed

Preserved the evaluated PICO500 actor, critic, Adam moments and update counters
exactly; initialized new simulation/RNG/history explicitly. Added only the
declared independent foot-placement reward to the previous objective. The
two-update wiring smoke, nine focused tests and initial CPU identity check
passed before this run. The original encoder and base decoder remain frozen.

Main94600 EXIT0:500 additional updates,1,024,000 new transitions,4,000 new
optimizer steps; learner500→1000. Recorded elapsed1351.552214s. Final checkpoint:
`train500_v1/checkpoints/foot_precision_model_1000.pt`
SHA256 ac2f844894e3e53c364ae466966a64db556ee255c6ab72e736fe15b87adb19fe.

Audit14325 EXIT0 verifies all1,024,000 stored rewards and timeout bootstraps,
exact parent actor/critic/Adam import, Adam steps4000→8000 and unchanged frozen
weights. All seven adapter pairs, root conditioner, exploration and critic
change. Reward arithmetic maximum1.525879e-5, within the declared5e-5 check.
This establishes training execution, not successful motion learning.
Final verification61608 EXIT0: nine focused tests pass in23.60s; all new
experiment drivers and implementation files pass Ruff E/F. All jobs terminal.

## Full requested motion results

| Checkpoint | Case | Completed/requested controls | Source seconds | Leg RMSE, own executed source | Tracking |
|---|---|---:|---:|---:|---|
|600|002|1417/1417|13.34/13.34|0.191435rad|FAIL|
|600|003|1569/1569|16.38/16.38|0.197350rad|FAIL|
|600|008|461/1114|2.22/7.28|0.181125rad|FAIL|
|600|PICO|4107/6530|75.14/115.60|0.190101rad|FAIL|
|1000|002|1417/1417|13.34/13.34|0.218125rad|FAIL|
|1000|003|1104/1569|15.08/16.38|0.221970rad|FAIL|
|1000|008|590/1114|4.80/7.28|0.165085rad|FAIL|
|1000|PICO|1880/6530|30.60/115.60|0.160060rad|FAIL|

These per-run prefix metrics do not compare identical durations. All incomplete
runs end at the unchanged bounded joint-range preview, with no intermediate
physical-state resets. Walking008 is excluded from optimization but was already
development-evaluated; it is not a fresh generalization test.

Matched PICO30.60s, parent500 versus final1000: leg RMSE0.162737→0.160060rad;
arm RMSE0.275610→0.269828rad; root p950.354987→0.190386m; relative-foot
p950.135367/0.163627→0.127001/0.145065m. Those local improvements do not offset
earlier failure or meet foot fidelity. Parent500 completed69.34source seconds.
Checkpoint600 completed75.14s but had worse matched69.34s root/leg tracking.
No source cropping, checkpoint promotion or causal reward-ablation claim.

Final PICO failure approaches right-ankle-roll upper bound with q0.248943rad,
dq+1.347234rad/s, remaining margin0.012857rad. Root height0.703163m. Its entire
30.60s source stays below the training30cm root-error cutoff, unlike600, which
crosses that cutoff after4.90s. Root drift alone therefore cannot explain the
final checkpoint's stop. Guard removal is not a tracking fix; physical damping
on the real robot is not diagnosed by this simulation.

## Independent checks and honest limitations

600 audit67154 EXIT0:75,540 fresh2ms torque-replay steps,7,554 landmark vectors,
384 policy re-inferences exact.1000 audit/comparison/diagnosis64347 EXIT0:
49,910 fresh2ms steps,4,991 landmarks,384 policy re-inferences exact. Actual
prefix joint-range excess0, effort ratio<=1; maximum velocity ratios0.809272
and0.662218 respectively. These checks do not change tracking FAIL to PASS.

Actual training input check:19,968 sampled environment-controls have bit-exact
lower-body240 and original29 VR21 references. Missing proprioception slots stay
zero;3,833 consecutive non-reset history shifts are bit-exact. Desired root
velocity maximum difference2.980232e-7. The first, noiseless audit FAILED at
orientation. It is retained unchanged. A separate configuration-aware check
45202 EXIT0 shows discrepancies within the already configured±0.05 orientation
and gravity noise, not a new packing defect. No noise or tolerance was changed
in training/evaluation. Unsampled inputs and measured root velocity were not
independently reconstructed. Both audit files and the failure diagnosis remain.

Training first/last100-update windows: mean ankle world RMS0.138272→0.142533m;
median completed reset interval97→102controls. Populations and sampled source
mix change, so these are not fixed-state or full-motion tracking comparisons.
Do not extend this recipe merely because its average episode reward is finite.

Saved-state video58816 EXIT0 renders every measured final PICO state. ffprobe
independently decodes1881frames,50fps,960x720,37.62s including7s entry/standing.
20s and37.5s frames inspected. This is a FAILED simulation, not robot footage:
`eval1000_v1/pico/FAILED_TRACKING_SIM_pico_checkpoint1000.mp4`
SHA2563306388f05f69a26d258d2a90a04bc9bcf9e2b8943c4509d8d04037460263fa4.

## Next decision boundary

Reject both checkpoints for deployment; preserve the earlier500 as evidence,
not a qualified fallback. No further500-update reward-only loop is scheduled.
Any next controller change needs a new bounded hypothesis targeting measured
leg/foot dynamics and predictive-limit admission while preserving full source
timing and original29 hand/head intent. Earlier range-only experiments already
failed to establish a dominant cause; do not repeat them unchanged.
Next read-only investigation: reconstruct ankle contact/torque behavior before
the final1000 stop from saved physics, distinguishing control request from
contact/coupling forces. This has not yet been executed for checkpoint1000 and
is not authorization to remove a guard or repeat old range-penalty training.

Full-motion fidelity, complete standing return, contact/foot-slip quality,
actual-clock timing and pause/disconnect/recovery still require qualification.
These clips use paired PICO/optical full-body references, not sensor-only live
PICO inference. Live world-state estimation, headset capture and supervised
Unitree handback remain separate unproved deployment requirements.
