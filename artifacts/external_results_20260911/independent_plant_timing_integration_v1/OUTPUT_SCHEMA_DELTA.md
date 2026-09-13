# Additive owner output fields

The producer output schema is unchanged from the reviewed integration: `timing_spans.bin`, `timing_gc.bin`, `timing_metadata.json` and `timing_owned_partial.json`, each retained in the main output manifest when written. Producer report field `timing_instrumentation` names successfully owned sidecar hashes/byte counts, explicit errors, hook status and completeness. Durable epoch/exit stages keep their existing compact probe counters.

The saved diagnostic verdict and `owner_completion.json` add identical `timing_sidecar_accounting` objects: `evidence_accounted`, `instrumentation_complete`, `declared_sidecar_files`, `unclaimed_partial_files`, `missing_sidecar_files`, `preservation_errors`, `probe_complete`, `independent_sidecar_math_pending`, `new_native_reads=0` and `verification_return_or_native_credit_inferred=false`.

This is identity/completeness accounting only. A failed or incomplete timing sidecar may be honestly accounted while diagnostic qualification remains false. Missing final preservation stage or unknown raw exit still prevents complete owner accounting under the original checks. No full clock scope, physical success, verification return or native credit is inferred from these metadata fields.
