# Prepared actual expert label collector

Preparation only. Neither collection, inference, compatibility execution nor fitting has run. Parent must first independently qualify the complete resumed1569-control lifecycle and separate250-control hold.

The planned final input is sibling `student_actual_oracle_control1_resume1001_v1/nominal/trace.npz`. The collector binds the original reviewed v2 receipt, process-resume receipt, immutable1001-control checkpoint hash and final trace through an explicit root qualification receipt. `qualification_TEMPLATE_NOT_AUTHORIZED.json` documents its fields and deliberately does not authorize execution. Both independent physical reports and both intent/quiet reports must pass and bind their exact traces before any actor inference occurs.

`source_snapshot/collect_actual_branch_labels.py` restores no simulation state and performs no physics or optimization. It rebuilds the measured sensor history, retains the real student's combined preclip action0 as provenance, and emits only controls1 through1268 as1268 expert labels. Original student control0 and terminal BFM controls1269 through1818 are excluded. The14 original pilot source files, including `LinearFeatures`, `infer_base`, `GoalFeatures` and observation helpers, remain byte-identical. The new artifact has its own source/input receipt.

Collector output uses the existing dataset schema plus named history/integration evidence. Previous action, state and flat history at query control1 must match the final20000 student's saved input exactly; base target and feature parity is measured with a1e-5 numerical bound. Native target limits and spans are checked exactly. Existing normalization is copied unchanged from the original training distribution. This is data preparation, not a model fit.

`source_snapshot/audit_label_compatibility.py` operates on saved arrays only. It calculates all1268-by1269 normalized feature RMS distances by explicit float64 differences, avoiding subtractive norm identities. Exact float32 duplicates, including signed-zero equivalence, are handled separately; zero-distance sensitivity ratios remain undefined in saved NPZ instead of dividing silently. It retains the full distance matrix, nearest and outside-time-neighborhood pair indices/raw discrepancies, same-control differences, phase summaries, command slew and residual/action ranges. Named worst joints and exact conflict groups are reported. No target averaging, smoothing, deletion or fitting is performed.

After root qualification and source review, collection command uses pinned WSL Python with BLAS/ORT one thread:

`python source_snapshot/collect_actual_branch_labels.py --qualification /path/to/root_qualification_receipt.json`

Only after collection succeeds:

`python source_snapshot/audit_label_compatibility.py`

Neither command authorizes training, a second query, new physical rollout or hardware actions. The parent reviews the resulting compatibility evidence before choosing any fitting step.

Use final `source_snapshot_v2` for both commands. Original unexecuted `source_snapshot` and `collector_frozen_inputs_v1.json` remain preserved. Read-only review confirmed all16 source/22 input hashes and original14 pilot files, then requested stronger audit provenance and separate conflict labels. The v2 audit verifies every frozen input, binds the collector receipt, requires exact controls1..1268/source frames12..1279, and reports exact-duplicate target conflicts separately from residual-label conflicts. Current receipt SHA2560dd8902c46b84fc0cfbd12bbaec8e5b1a2875c5470171c29f6b4c954819c6ad3. No collection, inference, compatibility execution or fitting has run.

Root qualification receipt476408e209a598adafba06a1c1d7ebda715543005ade45ae4e6069c5bb3ee86f authorized the single collection after independent original1569-control lifecycle/source/quiet and separate250-control hold passed. Reviewed v2 collector and compatibility audit completed2026-09-11T03:33:49UTC through durable hiddenPID21624, exit0. Outputs are `labels` and `compatibility`;1268 new labels SHA256e93753597450881c08a46fd520113abd21470653083dba0226c6450469d16ad1. Query-control1 state/features/base/prior/history all match the failed final20000 learner bit-for-bit. No exact duplicate/conflicting-feature groups were found, but near-input command sensitivity remains. No model fitting or new physics occurred. `NEXT_FIT_PROPOSAL.md` describes one controlled aggregate-data experiment, still unexecuted pending parent authorization.
