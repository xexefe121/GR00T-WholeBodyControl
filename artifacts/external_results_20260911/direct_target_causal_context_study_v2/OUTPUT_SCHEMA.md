# Proposed immutable output schema

All output creation remains gated on a concrete reviewed request and explicit fit selection. Existing artifacts are read-only. Outputs use this fresh v2 study root. The original v1 failure and consumed1437 initial forwards remain historical evidence.

Condition reports additionally state training_first_layer_execution='split_contiguous_1000_plus_323' and export_first_layer_execution='monolithic_float64_1323'. Restoration drift records stored_first_layer_width1323, unchanged original1000 contraction dimensions and the split execution identifier. Existing forward counts count complete model calls, not internal GEMM operations; each GPU32 first layer now performs two contractions. All numerical thresholds and other output schemas remain unchanged.

`shared/normalization.npz`: feature_mean/std float32(1323), original_feature_mean/std float32(1000), context_mean/std float32(323), context_mean64/variance64 float64(323), original joint_span float32(23), default_q float64(23), joint_limits float64(23,2). Old1000 fields must be byte-exact.

`shared/nominal_context.npy`: float32(9904,323), prior23 then incoming history300.
`shared/center_context.npy`: float32(3057,323), exactly nominal_context[center_to_nominal].
`shared/physical_context.npy`: float32(3054,323), verified actual-normalized applied prior then advanced_history.
Full-state endpoint context is center_context[row//116]; do not save an ambiguously shifted copy.

`shared/data_identities.npz`: unchanged nominal dataset/control/phase/source_frame, center_to_nominal, physical_successor/dataset/control, plus unchanged full58 metadata and axes. `context_alignment.json` binds all source hashes, all row/lag/prior comparisons, physical clipped-mask counts and the context arrays. `schedule_centers.npy` int32(3000,864), `schedule_axes.npy` int8(3000,864), exact old schedule first3000. `shared/output_manifest.json` binds every shared file.

`blinded/` and `causal/` each retain:

- `initialization.pt`: expanded actor_state (six tensors; first weight256×1323), fresh empty optimizer_state, restored source RNG, identical1323 normalization, source65000 PT/hash and condition. Normalized-zero masking is an input condition, not a different network.
- `restoration.json`: original1000 weights/normalization exact, all other parameters exact, extra columns zero, fresh moments, restored RNG, source subject and initial preclamp drift versus saved65000 diagnostics. The `causal` directory additionally contains `initial_comparison.json`, comparing both saved initial conditions before causal updates, with no additional forward.
- `used_centers.npy` int32(3000,864), `used_axes.npy` int8(3000,864); uncommitted rows sentinel−1. `training_progress.npy` float64(N,6), columns nominal/full_state/physical/weighted_total/LR/preclip_gradient_norm; N is the committed optimizer-row prefix. Cell curves float64(N,15), (N,54), (N,9).
- Five backends × three corpora as float32(rows,23) `BACKEND_CORPUS.npy`; rows9904/354612/3054. BACKEND names initial_GPU32/final_GPU32/CPU64/GPU64/ORT64. Matching metric JSON, call ledger, parity JSON and nine float64 drift arrays.
- `student_head.pt`: ordinary68000 actor/optimizer3000/RNG/norm/context condition/data/request/counters; `optimization_completed.json` written before export. `student_head.onnx`: publicfloat32 input(?,1323), output(?,23), internalfloat64; no model tracing calls. `promoted_parameters.json` verifies exact promotions.
- `report.json`: completed, optimization_completed, final_export_diagnostics_completed, numerical_gate_passed/export_parity_passed, condition, ordinary_final_step68000,additional_updates3000,optimizer_step3000,features1323,context_features323,fixed_full_state_coefficient,zero_calibration_calls,actual row/call counts, PT/head/norm/request/frozen/shared-manifest hashes. `output_manifest.json` binds saved evidence.
- On failure: `failure.json`, `failed_active_evidence.npz`, `failed_state.pt` (resumable=false), exact partial diagnostics/schedule/loss prefixes. Preserve optimizer_call_started/returned/synchronized and all six optimizer step counters. No automatic retry.

On a completed pair, `paired_report.json` records both ordinary conditions, shared initialization/schedule/normalization equality, four aggregate objective deltas and initial/final comparisons. Complete cell metrics remain in the two condition reports. It does not select a checkpoint or controller. A failure instead writes `paired_failure.json`, preserving the failed stage, completed-condition list and every existing condition report; no successful-pair claim is made. Parent process output records raw exit, diagnostic completion verdict, postrun pins and exact captured PID absence. A failed first condition stops the selected stage; the second is not silently skipped and called a completed pair.
