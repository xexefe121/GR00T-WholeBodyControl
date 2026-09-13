# Root source review guide

Eight snapshot files, two byte-exact qualified copies:

- `normalization_math.py`: independently implements the actual public-f32 -> f64 Cast/Sub/Div prelayer algebra; verifies literal exporter AST without importing it.
- `consistency_math.py`: six streaming alias views, member/anchor target differences, and 72 fixed-query/256-candidate proximity. No all-pairs matrix, no automatic label repair.
- `saved_inputs.py`: reconstructs exactly 9,904 nominal + 3,054 physical + 1,018 new rows, preserving actual applied prior, history, phase and target conventions. Loads no full58 probe expansion.
- `admission.py`: actual old fit lineage plus completed collector/request/source/qualification/output chain. Root's requested minimal gate is used; no `collection_review` artifact exists.
- `run_consistency.py`: mandatory concrete request/clearance before actual array loading, exclusive output directory, partial evidence preservation, post-read pin verification, no automatic retry.
- `test_consistency.py`: synthetic collisions, signed zero, cast equivalence, normalization mutations, bounded nearest-neighbor ties/phase filtering, actual fixed-count synthetic assembly, and metadata corruption rejection.
- `direct_contract.py`: copied exactly from qualified width81000 source; only its constants and original normalized-label cast are consumed.
- `input_schema.py`: copied exactly from qualified collection source; header checks reject even unused object metadata before deserialization.

No actual consistency request, collection array, checkpoint, model/ORT/runtime, optimizer or native call is part of this preparation. Synthetic tests use fake numeric arrays and fake metadata. The source-only exporter AST check reads the actual qualified Python text. Old fit JSON metadata schemas were inspected without numerical array/checkpoint reads.

Actual recovery collection is still conditional on all prior physical/intent/main/hold gates. An eventual diagnostic can report conflicts while completing successfully; root decides the next training design from the evidence.
