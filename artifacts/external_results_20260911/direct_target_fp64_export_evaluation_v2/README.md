# Same55000 checkpoint, separately qualified higher-precision export: version2

Source preparation only. No new head, witness, policy rollout or physics call is
produced here. The original 55,000-step fit completed optimization but failed its
FP32 export parity gate. Its report remains completed=false, numerical_gate_passed
=false and export_parity_passed=false. This package does not rewrite or override
those flags. A separately qualified export of the same checkpoint needs a new
positive release chain before this evaluator can call it.

The new export is owned by direct_target_fp64_export_v1. Its public interface is
float32 features[batch,1000] to float32 normalized_target[batch,23]. Normalization,
linear accumulation and the ELU expression execute in float64 before the final
float32 cast. The selected expression is
`Where(x>0,x,Exp(Min(x,0))-1)`. The exporter owns graph correctness and validation;
this evaluator does not re-export, fit or select a checkpoint. No extra runtime
dependency is installed. The existing WSL ORT runtime must successfully execute
the actual, separately selected one-call witness before the policy trial.

Version1 is preserved unchanged. Version2 changes only the new release helper's
root-audit schema handling; all30 other version1 modules remain byte-identical.
Twenty-nine original runtime modules remain byte-identical to the reviewed55000
preparation. Only evaluation_gate.py's release validation changes, using the new
pure export_release_gate.py. Direct features, startup and terminal BFM behavior,
learned applied-action feedback/history ordering, proposal conversion and native
strict checks remain unchanged. The mixed trace delta field retains its existing
float64 promotion once learned rows are present.

The controller still computes default_float64 + existing_span_float32 promoted to
float64 * output_float32 promoted to float64, then clamps against the original
float64 native bounds. It uses no BFM actor/backward calls during the learned
phase. Startup controls0..249, learned250..1268 and terminal BFM from1269 are
unchanged. The canonical request remains1569 controls followed conditionally by
250 continuous hold controls. The original full291 initial fixture,250-control
BFM prefix, query250 feature/history/prior checks and exact first-head witness
comparison remain mandatory. No clock foundation or filter is connected.

## New release chain

The original fit report, failed source ONNX, same checkpoint, normalization,
training request and training manifest are individually pinned. The original
root audit must report evidence_audit_passed=true, export_qualified=false and
canonical_cleared=false. All six original role paths and hashes must be explicit
members of its actual input_sha256 map. Windows path separators and case are
normalized without accepting a matching digest from a different path. A separate
training review directly binds the exact root audit and all six role hashes.
A dataset review directly binds the
immutable training manifest. These receipts replace repeated traversal of
unused training corpora; they do not claim the original export passed.

The new export report must identify those same original subjects and its new
head, report zero optimizer/BFM/native work and pass the unchanged1e-5-radian
preclamp target parity tolerance. Completed export owner evidence must bind the
new request, manifest, report and model along with the original subjects, confirm
known zero exits, unchanged inputs and stopped processes. A new export review
must directly bind that chain plus positive training evidence. The source review
must bind both the unchanged driver/witness and new release-gate source.

Direct release subject maps use `direct_subject_sha256`. Training subject keys
are fit_report, checkpoint, source_head, normalization, training_manifest and
training_request. The training review adds root_training_audit. Export subject
keys are head, checkpoint, fit_report, source_head, normalization, export_report,
export_request and export_manifest. The export review adds root_training_audit
and export_owner_completion. The immutable root evidence audit retains its actual
input_sha256 schema; no nonexistent direct_subject_sha256 field or additional
bridge receipt is required on that audit. An unrelated digest elsewhere in a
receipt cannot satisfy these named subject requirements.

## Source checks and intended workflow

The freezer accepts only actual existing artifacts and real selected review
configuration. There are no placeholder hashes or empty success receipts. It
validates the new release before creating a binding. After actual release review
and selection, prepare the one-call WSL witness binding and durable hidden
launcher, get the concrete launcher review and run once. Only a completed,
byte-matched witness permits the separate canonical binding. Any failed or
incomplete physical run keeps a nonzero diagnostic outcome even if Python exits
zero. Unknown exits and partial failures remain preserved; no retry is automatic.

The existing explicit runtime inventory is reused with the new source directory.
It pins the actual native/reference/BFM/head/ORT reads and immutable qualification
receipts, excluding unused training corpora and CUDA trainer packages. This is a
source-level package inventory, not a measured syscall trace or wall-clock claim.
The existing WSL standard library and system libraries remain the runtime trust
boundary.

Synthetic checks cover the unchanged30 controller tests,21 launcher/accounting
tests and33 release checks. One old helper test was adapted because subject
validation moved from the freezer's generic role loader to the explicit release
gate; its initial failing compatibility receipt is preserved. The new tests
verify original failure preservation, exact subject mapping, unchanged parity
threshold, positive training evidence, new-export completion and stopped owner
processes. They execute no model, optimizer, native step or dependency install.
