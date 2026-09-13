# Reference-path cost fixed; timing improves, deployment still fails

Goal remains ACTIVE. These are measured runtime changes, not policy/foot-tracking
improvements. All twelve timed attempts, including four timing failures, remain
saved. No physical robot command, policy promotion, source retiming or weaker
joint/effort/freshness/deadline threshold.

## Implemented

`gear_sonic/teleop/kinematic_reference.py` prepares two independent reference
scratch states, reuses them, and computes reference body transforms with
mj_kinematics. It removes per-packet MjData allocation and two unnecessary full
dynamics/constraint solves. Actual native23 integration is unchanged. All1817
packets from existing walking002/003/008 produce identical fields. Isolated
conversion p95 changes1.900→0.708,1.874→0.701,1.770→0.637ms respectively.

`gear_sonic/teleop/cpu_paced_inference.py` separately disables ONNX worker
spinning for encoder, decoder and balance sessions. Default thread count, graph
options, model bytes and precision remain unchanged. This follows a testable
pool-contention hypothesis from the [ORT threading documentation](https://onnxruntime.ai/docs/performance/tune-performance/threading.html),
not an established explanation of every OS outlier. Earlier inference-only
microbenchmarks were insufficient to judge actual full-loop schedulability.

All656 captured encoder/history inputs produce exactly identical raw23 and
decoder994 arrays. Complete656-source+250-balance virtual replay has bit-exact
qpos/qvel/time versus the measured matching baseline. The parity preflight warms
its own inference sessions by testing them; each timed case constructs fresh
sessions, with NO added runtime warmup.

## Fixed six-case timed matrices

Prepared FK with default spinning:3/6 scenario checks pass. Nominal full-walk
SONIC counts19/656/47 of656. Pause fails early at150SONIC controls; gap and
malformed payload both correctly latch balance at200. No favorable-run selection.
The first failed nominal control takes22.854ms, including17.589ms inference.

Prepared FK with non-spinning sessions:5/6 checks pass. Nominal SONIC counts
656/212/656; pause/gap/payload all correctly latch at200. Two full walking runs
then execute the complete250-control EOF balance tail. All cases integrate
906controls/18.12s virtual physics, with no pose resets after initialization.

The remaining no-spin failure is NOT slow inference alone: at control211 the
process begins18.674451ms late, then uses12.274717ms (inference7.910008ms,
reference conversion0.942791ms). It crosses the unchanged next20ms deadline,
latches balance, and reports failure. Maximum execution over no-spin cases is
14.232350ms; meeting the compute-time budget alone does not erase wake lateness.
No-spin nominal cases1/3 maximum execution12.650/13.245ms. Full matrix still
FAILS; do not declare wall-clock qualification based on the two successful runs.

## Independent evidence

`audit_matrix.py` verifies107/110 source pins per case, sender/receiver hashes,
exact timing and fault arithmetic, source ages<=100ms, monotone latched balance,
and exact same-policy SONIC prefixes. Fresh native23 integration of every saved
2ms torque matches all substep joints and every control qpos/qvel bit-exact.
Total108,720 independently replayed physics substeps across twelve cases.
All actual range excess0, effort ratio<=1, maximum velocity ratio0.569275.
No contact/slip, torque-slew, headset or physical handback qualification implied.

Original full walk tracking remains unchanged: leg RMSE0.207371rad,
pelvis-relative foot p950.266102/0.208334m, root p951.056420m. Runtime changes
cannot fix this policy failure. Earlier115.60s PICO momentum variants remain
rejected separately; this walking matrix is not their test or their promotion.

Scoped runtime tests pass; final exact count and command completion are recorded
in PROGRESS.md. Production modules, new matrix drivers and auditor pass Ruff
E/F. Five overlong lines in the immutable executed preflight.py are preserved
rather than altering its bound source after execution. An initial matrix setup
attempt rejected an artifact-path source-closure seed before output directory,
publisher or physics creation; corrected seed uses the actual gear_sonic modules
and binds the experiment driver separately. No failed controller trial was
discarded. Initial unit fixture errors were fixed before preflight execution.

## Reproduce and use

An opt-in SIM-only localhost receiver now exists:
`python -m gear_sonic.scripts.run_g1_true23_prepared_paced_sim --help`.
It uses the same prepared FK/non-spinning settings and existing50Hz runtime.
Its exit0 additionally requires full requested source completion; completing a
balance tail after lost input cannot masquerade as full recorded-motion success.
This is NOT a hardware launcher or promoted live-headset controller.

Typical arguments, with an already-running same-host saved-packet publisher:

```bash
cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:/mnt/z/codex/GR00T-WholeBodyControl/external_dependencies/unitree_rl_mjlab \
  /root/.venvs/g1_true23_mjlab/bin/python -m gear_sonic.scripts.run_g1_true23_prepared_paced_sim \
  --repository-root /mnt/z/codex/GR00T-WholeBodyControl \
  --encoder-report artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.encoder.json \
  --decoder-report artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.decoder.json \
  --endpoint tcp://127.0.0.1:PORT_FROM_LOCAL_PUBLISHER \
  --source-controls 656 --tail-controls 250 \
  --output-directory artifacts/NEW_UNUSED_SIM_OUTPUT
```

PORT_FROM_LOCAL_PUBLISHER and output name are explicit placeholders. Endpoint
must be localhost; timestamps must use the consumer host's monotonic clock.
For the tested self-contained sender/consumer workflow, read run_matrix.py and
run_matrix_nospin.py. They refuse existing saved output directories; preserve
these results and use a new versioned output for any further experiment.

Evidence SHA256:

- Default matrix:2b3fe58c91167f193b871ac18ccd164824c0dfdf87444c1c941f1eb539a241b3.
- No-spin matrix:d028892e27f08a356a1f8834c89d600c2a6a95fafe226e4e89d78c24f045df9b.
- Default audit:31e4f0b01d66f5e31850aadf8f4483cd4adee120e2a3b23ea106f8efd3a1d510.
- No-spin audit:8cbd53ceacf8ce6f0bd05b265c68ea33df63d7d652ad882b7412d6b7db03b825.

Next work must address policy fidelity and the observed scheduling jitter.
Another identical run to obtain a green result, weakening20ms checks, or
claiming that a headset connection alone unblocks deployment is not justified.
