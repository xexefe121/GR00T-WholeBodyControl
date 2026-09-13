# Private unguided seed-scoring optimization

Eight bounded comparisons passed bit-for-bit on the saved actual PICO control-3805 state. The private adapter computes one identical unguided MuJoCo lane and broadcasts its integration/derived outputs, retaining all nine public output lanes, original nine-lane cost arithmetic, and nine-lane guided search. No shared source, running controller, model, cost, horizon, target, or solver parameter changed. No full iLQR solve, policy inference, training, or controller trial ran.

Five alternating paired CPU1 timings on the same H30 seed:

| Seed scoring | Original 9 lanes, median | Private 1 lane, median | Saved per call | Speedup |
|---|---:|---:|---:|---:|
| Existing soft path | 314.86 ms | 48.32 ms | 266.54 ms | 6.52× |
| Every-2ms hard checks | 510.92 ms | 143.79 ms | 367.13 ms | 3.55× |

Timings include copying/expanding outputs and unchanged cost/feasibility work. They exclude planner construction and file I/O. Pinned MuJoCo 3.2.3, NumPy 1.26.4, Intel i7-10870H, one Batch/BLAS/OpenMP thread; the process was not pinned to one core. The host was concurrently running the authorized full PICO job, so this is not a quiet timing measurement. The earlier profile used eight Batch threads and a different state. These savings cannot be directly subtracted from that full-solve profile. No end-to-end expert speedup or real-time qualification is established; derivative work remains unchanged.

## Exact evidence

`report.json`, `source_binding.json`, and `validated_v4/` contain the final source/input hashes, outputs and witnesses. The frozen current-run core/tracker/model copies are respectively `1efc9cb2…`, `7695d037…`, and `61d1c168…`. Imports were checked to come entirely from the private snapshot tree.

Comparisons cover soft and hard positive seed scoring; a subsequent nine-lane guided call with zero gains; returning to unguided after that guided call; the existing physically infeasible original warm seed; and a NaN first target rejected before any physics step. For all eight, complete returned states, targets, costs, feasibility records, full `mjSTATE_INTEGRATION`, and every bound qpos/qvel/ctrl/warm/time/warning/force/body-pose field match byte-for-byte. Each timed pair's full states/targets/cost arrays was also checked.

The positive hard seed has cost 1137.2608274877853, zero joint excess, maximum native speed ratio 0.821271528300455 and effort ratio 1. The original warm seed is rejected in every lane at control 28/substep 5 with the identical joint-range witness. The NaN target is rejected at control 0/substep 0 with the identical nonfinite-target witness. These are existing planner rollout semantics, including its per-control warmstart reset; no new full-native-continuation certificate is claimed.

## Important Batch contract finding

Naively copying only visible poses is insufficient. Even copying the full integration vector plus visible buffers is insufficient: mjbatch also keeps private mirrors of input fields. Dormant lanes' mirrors remain stale. A later assignment such as `qacc_warmstart[:] = 0` or `time[:] = 0` can equal a stale mirror and therefore be ignored, leaving a copied nonzero value in the authoritative integration state.

The final private adapter applies the same qpos, qvel, ctrl, zero-warmstart and hard-clock overrides directly to the authoritative integration vector at the time a step consumes them. Overrides are deferred when all lanes are rejected before a step, preserving existing deferred-write semantics. Failed prototypes, sources and partial outputs are retained in `attempt1_sources`, `attempt2_sources`, `attempt3_sources`, `validated`, `validated_v3`, and `initial_comparison_erratum.txt`.

This is a bounded proof of potential, **not production-ready integration**. Only one saved state and the listed guided-zero-gain probes were tested. Mixed valid/invalid guided lanes with nonzero feedback, per-lane interventions, and arbitrary bound fields were not qualified. In particular, the private input synchronization currently writes all integration rows when at least one guided lane remains active; a production version should preserve inactive-row deferred-write semantics too, or use a separate one-lane Batch to avoid dormant-mirror coupling. No further run or source change was made for that broader case. Keep this adapter out of the active full-source job.

The safe next implementation decision is either a dedicated single-lane seed pool with an explicitly narrow output contract, or a carefully scoped Batch synchronization primitive verified on mixed guided lanes. The measured potential is useful for offline expert throughput; it does not close the real-time gap.
