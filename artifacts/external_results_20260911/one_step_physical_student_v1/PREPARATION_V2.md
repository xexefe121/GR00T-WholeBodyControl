Preparation remains unselected for fitting. Use `source_draft_v2` for the next review; v1 and its exact evidence remain preserved.

Reviewer requested two diagnostic corrections, both implemented without changing the objective, model, optimizer, RNG, normalization, inputs or call counts. The first24 physical metric now uses the fixed requested successor-clock window and reports its valid/failed coverage. Each diagnostic graph call clears earlier outputs and records its current stage, batch and returned flag, so a later-batch fault cannot pair current inputs with an earlier result.

All32 original helpers and the derived trainer are unchanged. Only the new physical diagnostic helper differs from v1. `preservation_v2_derivation.json` records exact changes and source hashes. All32 source/synthetic/stub tests pass, including a second-batch exception and an early-failure window. No real model/optimizer/ONNX/native calls ran.

The collector now has a selected prefix-preserving continuation at `one_step_policy_branch_collection_resume2969_v1`; final outputs and counters are still pending. The trainer has no hardcoded collection destination and no final input request. Final freeze must bind the real completed manifest/report, original failed-prefix lineage, validity mask, root independent data/conflict audits and source review. No launcher or fit authorization exists.
