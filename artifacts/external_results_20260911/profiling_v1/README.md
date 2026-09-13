# One exact PICO planning-state profile

The existing H30, five-iteration, eight-thread solver took **2444.383 ms** at saved control **3200**, versus a 100 ms commit interval. Dynamics finite differences consumed 60.1%. This is one isolated, instrumented planning call, not a timing distribution or a live-control qualification.

Control3200 was chosen because its immutable checkpoint preserves the complete prior 30-target warm horizon. No corresponding checkpoint3700 was found. The actual qpos/qvel and all preceding BFM observation/action history match the frozen producer. The existing planning block was extracted with identical AST statements; instrumentation delegates to its frozen functions. Shared repository sources were not edited.

| Stage | Time | Share |
|---|---:|---:|
| Dynamics linearization, five evaluations | 1469.052 ms | 60.10% |
| Five guided line-search rollouts | 388.132 ms | 15.88% |
| Fresh BFM target generation | 243.573 ms | 9.96% |
| Recorded/shifted/fresh seed scoring | 223.067 ms | 9.13% |
| Cost expansion, five evaluations | 68.262 ms | 2.79% |
| Backward pass, five evaluations | 51.007 ms | 2.09% |

Within fresh generation, actor inference took155.538 ms across30 calls; backward-network inference10.405 ms; native private physics24.517 ms across300 steps. The five dynamics linearizations each evaluate30 knots ×82 lanes ×10 native steps: **123,000 native2ms steps total**. Native batched stepping consumed1441.014 ms of the1469.052 ms linearization stage. Python assembly and copying are not its dominant cost.

The separately replayed **existing checkpoint serialization** workload took2.298 s for the exact saved3200-control archive, including compression, flush, fsync and atomic replace. Its arrays were independently read back and compared exactly. This is outside the producer's solve timer and does not occur every100ms plan. Saving this profile's extra diagnostic arrays was also timed separately. No physical control or checkpoint chronology was resumed.

## Exactness and evidence

- First five returned planned states, targets and feedback gains are bit exact against the existing producer's saved plan3200. Final cost is exactly166.34450081817036. The old uninstrumented producer logged2584.817 ms at that state on its earlier host load; this is not a controlled speed comparison.
- Selected seed remains `shifted_mpc`: recorded cost1991.467189697736; warm cost406.041820642801; fresh cost1335.9719759216327.
- Full new horizon outputs are saved: states31×59, targets30×23, gains30×23×58. Original uncommitted full horizon was not archived, so only the available first-five outputs and scalar cost have independent producer parity evidence.
- `solve_arrays.npz` also preserves returned seed, derivative, backward and line-search arrays. All nine unguided lanes have bit-exact states, targets and costs for **all three seeds**.
- `events.json` contains1190 nested timed calls. Exclusive accounting sums exactly to total time; stage table avoids overlapping parent/child totals.
- `report.json` binds original source/model/reference/checkpoint hashes, runtime identity, frozen imports and arrays. `analysis.json` derives the table and duplicate-lane checks from those saved artifacts. `runtime_library_supplement.json` independently confirms native `mj_version()==323` and actual libmujoco3.2.3 hash.

## Equivalent optimization recommendation

Replace **only unguided seed scoring** with a one-lane rollout. Without gains, alpha is unused: each of nine lanes starts from the same state and receives the same clipped targets. The measured saved arrays confirm all nine lanes are identical here. Preserve lane0 arithmetic, warmstart resets, forward/feature timing, target clipping, cost summation and warning-delta rejection. Keep the genuine nine-alpha guided line search unchanged.

This optimization was **not implemented or benchmarked** by this task. Even eliminating the entire223ms scoring stage for free leaves2221ms, over22 times the100ms commit budget. Reducing nine lanes will not provide a ninefold wall-time reduction because current lanes run across eight workers. It is a small exact optimization, not a route to a real-time claim.

Do not substitute the fresh BFM generator's private trajectory for its scored rollout without separate numerical validation: private continuous manual PD and the scorer's affine PD/warmstart-reset sequence have different computational paths. The existing core already skips its unused last derivative refresh and reuses the selected initial rollout; those are not new opportunities.

## Runtime and scope

Pinned Python3.11.15, NumPy1.26.4, MuJoCo3.2.3, ONNX Runtime1.23.2; Intel i7-10870H, eight physical/sixteen logical CPUs. OMP/OpenBLAS/MKL threads1; each mjbatch pool has eight workers; process had17 threads. `/proc/stat` suggests roughly5.87 busy logical CPUs averaged over this plan, including any other WSL work. Windows preflight load was4%; no other team heavy job ran during this solve. WSL load averages and raw CPU counters are saved. New artifact storage is E-backed; preflight free space was E225GB, C6.56GB, Z780MB.

WSL E: was unmounted after restart; it was mounted locally as drvfs. Runtime paths were unchanged. Native library, controller costs, H30/commit5/iterations5, original source timing, finite-difference epsilon1e-6, all-joint margin0.05/weight2000 and relative-foot weight400 remain unchanged. Source/reference preview remains offline740ms packets/up to760ms raw pose support. No robot, Pico, DDS, deployment, controller execution or new physical-quality claim.

The runner writes beside itself. To repeat later, copy `profile_one_saved_plan.py` and `source_snapshot/` into a **new** output directory first; never rerun it in this preserved evidence directory. Launch that new runner with the pinned WSL Python and the same environment variables. One solve was sufficient here; no uninstrumented repeat or parameter sweep was run.
