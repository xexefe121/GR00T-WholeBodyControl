# One controlled aggregate-data fit proposal

Status: proposal only. Collection and saved-array compatibility completed; no model fitting, learner inference, new expert query or physics test has run as part of this proposal. Parent chooses whether to authorize it after independent data review.

The qualified expert continuation supplies1268 new applied-target labels at controls1..1268, including the final20000 learner's actual divergence state. Query-control1 features, base target, state, prior action and history are all bit-identical to that learner's saved input. Its previously recorded prediction differs from the new expert by0.089792rad joint RMS, with left ankle pitch0.233027rad low. This is a concrete missing supervised example, rather than evidence that more epochs on the old dataset alone would suffice.

Compatibility finds zero exact-feature duplicate groups and zero exact target or residual conflicts across both datasets. The16 nearest matches with normalized feature RMS distance below0.01 have target-difference RMS p95 of0.036583rad. There is still significant local sensitivity: new control35 versus old91 differs by only0.017427 normalized feature RMS while targets differ0.205739rad RMS; right ankle pitch differs0.499850rad. New146 versus old80 has right ankle pitch difference0.906815rad at feature distance0.023570. These are distinct measured inputs and qualified actual targets. They justify reporting fitting difficulty explicitly, with no label averaging, deletion, command smoothing or arbitrary output cap.

Proposed single experiment:

- Combine unchanged original1269 rows with all qualified new1268 rows. Keep original teacher control0; exclude new student control0 and all terminal BFM controls. Freeze both dataset hashes and root qualification receipts.
- Preserve1069 features, goal offsets, original feature mean/std with0.05 floor, native joint spans,256/256 ELU architecture, linear residual output, squared normalized-residual loss, AdamW3e-4/weight decay1e-5 and gradient clipping10. Preserve all observation and combined preclip action semantics.
- Restore the final20000 model, AdamW state, Torch RNG and NumPy RNG from its existing checkpoint exactly; do not restart or pick an earlier model. Record exact restoration/parity evidence before new optimization.
- Use a256-row batch with32 old and32 new rows from each of the four existing phases: entry, acquisition, source and return. Sample with replacement as before. This gives equal old/new contribution within each phase despite the one-row entry difference. Change no other loss weighting.
- Use a fixed20,000 additional updates, ordinary final global40,000 checkpoint only. Log separate old/new whole-dataset errors, first24/per-joint errors and actual query-control1 error every1,000 updates. These are fit diagnostics, not validation or checkpoint selection. Nonfinite training aborts and remains a failed experiment.
- Verify Torch/ONNX export agreement and perform exactly one canonical full1569-control student rollout with the ordinary final checkpoint, followed by the existing continuous250-control hold only if the original lifecycle completes. Reuse the frozen runtime, source goals, native physical limits and terminal yaw4 switch. Record all actual2ms states/torques/clocks/warnings and precontrol history; no resets or shortened-success criteria.
- Parent independently audits the complete source, physical limits and quiet windows. An early failure remains a failure; preserve its actual state/history for diagnosis. Do not start another query, fit, architecture change or checkpoint selection automatically.

This remains a single-source offline behavioral experiment with future goal preview and simulation root state. It does not establish all-source robustness, received-only input, real-time deadlines or hardware readiness.

Evidence: `labels/report.json`, `compatibility/report.json`, `compatibility/cross_distances_and_pairs.npz`, and `compatibility/near_duplicate_witnesses.json`. New labels SHA256e93753597450881c08a46fd520113abd21470653083dba0226c6450469d16ad1. Original labels SHA2568003407282b01f2c66fe1720d7eb038f43377c1d6eab80dce47f1f75911e8083.
