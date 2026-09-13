# Saved full-state fit audit

Source preparation only. No actual fit audit has run. The auditor never constructs or executes a task model, recomputes a task gradient, updates an optimizer, imports ONNX Runtime, or steps MuJoCo.

It reconstructs the complete private PCG64 schedule (10,000 ×864 center/axis pairs), six fresh AdamW states, original55,000 weights/RNG/normalization restoration, final optimizer step10,000/global65,000, fixed learning rates and saved loss composition. Calibration checks use the three actual retained forward arrays and18 parameter-gradient arrays, including54 group losses, norm ratio, Gram matrix, cosines and first-order gradient geometry. Saved gradients are checked algebraically, not regenerated.

All five backend/corpus groups contain367,570 public-float32 predictions each. Independent calculations verify all15 nominal,54 full-state and9 physical cells, fixed first24 windows, target reconstruction, teacher clipping groups, all7,185 batch partitions, nine CPU64/GPU64/ORT64 preclamp comparisons and nine drift arrays with clipping-mask counts. The old independently reviewed graph helper verifies all20 ONNX nodes, exact promoted weights and public float32 interface without inference. Original graph, tree-comparison and base-math helpers are copied byte-for-byte.

Nominal metrics widen float32 squared errors before reducing and compare with an explicit3e-7 relative roundoff tolerance; remaining saved floating algebra uses5e-12 relative/1e-14 absolute tolerance. The release threshold remains exactly1e-5rad on preclamp physical target differences.

Numerical release failure is preserved: if optimization and all selected diagnostics completed and the saved numerical failure is internally exact, the auditor can return `evidence_audit_passed=true`, `export_qualified=false`, `passed=false`. It never changes the original report or selects another checkpoint. Unexpected evidence mismatch fails the audit and preserves checks/reason. Process absence remains a separate owner qualification; the auditor checks recorded start/child/exit PID linkage and exact pre/post pin maps.

The actual request is created only after completed fit evidence and root source review exist. It binds the13 literal release subjects, including corrected full-state data owner v2. Run once after those gates; no retry policy or model calls are embedded.
