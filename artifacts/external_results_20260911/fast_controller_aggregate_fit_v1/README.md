# One aggregate-data continuation

Prepared only; source and launcher require independent review before fitting.

The prior final20000 learner is restored with its full model, AdamW state, Torch RNG and NumPy RNG. Model tensors, optimizer tree, RNG, original normalization/spans, all1269 prior predictions and restored ONNX bytes must match before any update. The source does not reconstruct or reset the old fit.

Training combines unchanged original1269 labels with qualified fresh1268 actual expert labels. Student prefix0 and terminal BFM commands are excluded from the fresh labels. Each256-row batch draws32 old then32 new examples from each of the four existing phases, using explicit control indices. Fresh entry has249 rows. Original normalization, feature/observation semantics, native spans,256/256 ELU network, loss, optimizer settings and output clipping remain unchanged.

Exactly40,000 additional updates run at global steps20001..60000. Only the ordinary final60000 checkpoint is eligible for the single canonical test. No early stopping, best-checkpoint selection, intermediate physics, new query or fitting sweep. Diagnostics retain old/new/all/phase/first24 errors and predictions on fixed prior nearest-neighbor witnesses. Both first24 global-control windows and first24 dataset rows are labeled explicitly.

`run_fit_durable.ps1` is a hidden durable Windows launcher for the original Python3.10/Torch CPU runtime, one thread. It records PID, stdout, stderr and exit status. It only runs fitting. Final ONNX/Torch parity and model/report hashes must be reviewed before starting the separately authorized canonical1569-control learner rollout and conditional250-control hold.

The15 prior source files are byte-identical. `source_snapshot/fit_aggregate_once.py` is the only new trainer. The original frozen evaluator and runtime will be used for canonical evaluation: the new head produces control0 itself from canonical initialization, then operates on its own measured history. Existing native500Hz/outer50Hz physics and terminal yaw4 switch at1269 remain unchanged. No teacher targets or learner/expert prefix commands are replayed.

This single-source offline experiment does not qualify real-time teleoperation, received-only input, all-source robustness or hardware.

Prefit review cleared final `source_snapshot_v2` and `run_fit_durable_v2.ps1` in sibling `aggregate_student_prefit_review_v2`. Unexecuted v1 source/receipt/launcher remain preserved. Current frozen receipt1471e7fd28602e829eed1ac14bae9d3e8474f5a1bdaeeeedb6bdffbf0bdd3fb7. The one fit launched2026-09-11T03:46:44UTC, hiddenPID8636. Runtime restoration passed model tensors, full optimizer tree, both RNGs, normalization/spans, all1269 old predictions and byte-identical restoredONNX before any new update. No physics has run. The prepared `run_canonical_durable.ps1` requires a separately reviewed final-export clearance receipt and rejects existing outputs; it has not been launched.
