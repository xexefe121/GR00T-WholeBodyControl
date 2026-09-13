# Fixed full-state fit output schema

All paths below are relative to `fit/`. Source preparation only; actual fit awaits completed data and concrete review. Ordinary final global step65000, new optimizer step10000. The original55k weights, normalization and default/span/native bounds remain the initialization subjects.

| File | Exact shape/type or contents |
| --- | --- |
| initialization.pt | actor_state: six original float32 tensors; optimizer_state: fresh AdamW, empty state; rng_after_restoration: original55k Python/NumPy/Torch CPU/CUDA RNG; source_checkpoint_sha256; ordinary_start_step55000; optimizer_start_step0; fresh_actor_initialization=false; fresh_optimizer=true; feature_mean/std float32 tensors |
| normalization.npz | Byte copy of original normalization archive |
| data_identities.npz | dataset/control/phase/source_frame for9904 anchors; center_to_nominal(3057)int64; physical_successor/dataset/control(3054)int64; full_state_dataset/control(3057)int64; full_state_phase/cell(3057)int8; axis_group(58)int8; axis_radius(58)float64; axis_units(58)U5 |
| nominal_normalized_labels.npy | (9904,23)float32, explicit cast of original absolute labels normalized by promoted span |
| schedule_centers.npy, schedule_axes.npy | (10000,864)int32 and int8; fixed PCG64 seed20260911, ordered54 cells,16 pairs each; sampler.json binds both SHA and initial/final independent generator state |
| used_centers.npy, used_axes.npy | Same shapes/dtypes, initialized−1; committed row written only after optimizer return and CUDA synchronization |
| calibration_gradients.npz | nominal_0..5, full_state_0..5, physical_0..5; float32 shapes (256,1000),(256),(256,256),(256),(23,256),(23), exactly returned ordered autograd.grad tensors |
| calibration_forward_outputs.npz | Actual same-call nominal(9904,23), full_state(1728,23), physical(3054,23) float32; nominal_cell_losses(15)float32, full_state_cell_losses(54)float64, physical_cell_losses(9)float64; saved before gradient calls, SHA bound by coefficient.forward_outputs_sha256 |
| coefficient.json | coefficient=norm(nominal)/norm(full_state); float64 CPU norms, Gram, cosine matrix, combined-gradient norm/descent dots; loss scalars; schedule_index0; parameter_names; gradient SHA; attempted/returned/synchronized/verified3 gradients; actor/RNG unchanged, zero optimizer updates |
| restoration.json | Exact actor/norm/RNG; fresh optimizer0; exact initial nominal/physical saved GPU predictions. Old velocity overlap difference reported under changed full58 batch positions, no false byte gate |
| training_progress.npy | (10000,6)float64: nominal loss, unweighted full-state loss, physical loss, weighted total, LR, preclip gradient norm; on failure exact completed log prefix |
| nominal_cell_losses.npy | (10000,15)float64 |
| full_state_cell_losses.npy | (10000,54)float64, ordered dataset-phase cell then tangent group |
| physical_cell_losses.npy | (10000,9)float64 with original requested99/819/100 denominators |
| student_head.pt | Ordinary65000 actor_state float32, optimizer_state six AdamW10000, final rng, original feature_mean/std/span/default/limits/indices, global/additional/optimizer steps, full_state_coefficient and coefficient SHA, original source checkpoint SHA, request, counters |
| optimization_completed.json | Final checkpoint SHA and completed optimization; export validation explicitly pending at save time |
| initial_GPU32_{nominal,full_state,physical}.npy | Float32 outputs (9904,23),(354612,23),(3054,23) |
| final_GPU32_{nominal,full_state,physical}.npy | Same output shapes/types |
| CPU64_{nominal,full_state,physical}.npy | Final promoted arithmetic, public float32 outputs with same shapes |
| GPU64_{nominal,full_state,physical}.npy | Same |
| ORT64_{nominal,full_state,physical}.npy | Same |
| {backend}_metrics.json | All15 nominal,54 full-state,9 physical cells, first24, applied/preclamp target RMSE, full-state teacher clip flags; objectives and fixed weighted objective |
| diagnostic_calls.jsonl | backend, corpus, start/stop, returned/synchronized/verified per call. Separate corpus batch256 partitions:39+1386+12=1437 per backend |
| promoted_parameters.json | Final checkpoint SHA, each promoted buffer shape/type and float32 round-trip proof; internalfloat64/publicfloat32 |
| student_head.onnx | Manual20-node graph, no tracing; exact float32 source values promoted to64; native Torch64 ELU, ONNX bounded Exp/Min/Where equivalent; public float32 input/output |
| export_parity.json | comparisons keys `{corpus}_{left}_{right}` for CPU64/GPU64, CPU64/ORT64, GPU64/ORT64; nine preclamp maxima in rad; fixed1e-5 tolerance; passed/nonfinite flags |
| drift_{backend}_vs_final_GPU32_{corpus}.npy | Nine float64 physical preclamp target-difference arrays, same corpus output shapes |
| drift.json | Nine entries: max_abs_preclamp_rad, RMS_preclamp_rad, shape; old/new_clipped_rows/components and clipping_changed_rows/components |
| output_manifest.json | files mapping relative filenames→SHA; training_request_sha256/frozen_receipt_sha256; excludes mutable progress and self/report |
| report.json | completed/optimization_completed/final_export_diagnostics_completed/numerical_gate_passed/export_parity_passed; ordinary_final_step65000/additional_updates10000/optimizer_step10000; features1000/head_output normalized_target; fixed coefficient/metrics/counters, exact PT/ONNX/norm/request/frozen/output-manifest SHAs; same1e-5 gate; no FP32 ONNX release |

Counters are nested `calibration`, `training`, `gradients`, `diagnostics`. Forward counters retain calls_attempted/returned/synchronized/verified and rows_attempted/returned/verified. Gradient counters retain attempted/returned/synchronized/verified. Diagnostic labels are exactly initial_GPU32, final_GPU32, CPU64, GPU64, ORT64. Four Torch passes total1470280 rows/5748 calls; ORT367570 rows/1437 calls. Calibration14686 rows/3 forwards/3 gradients. Training146860000 rows/30000 forwards. Native/BFM/export tracing calls0.

On any fault: failure.json, failed_active_evidence.npz, failed_state.pt and available gradient partial arrays are saved best-effort. Metadata includes the active attempted global/update index, exact actual six optimizer counters and whether optimizer call returned/synchronized. An interrupted optimizer call is never described as a resumable checkpoint. No retry or inferred success from a process exit alone.
