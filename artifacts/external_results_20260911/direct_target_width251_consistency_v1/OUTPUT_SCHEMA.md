# Request and outputs

No actual request or output directory is created during source preparation.

The future request has `root_selected_saved_diagnosis: true`, `model_calls: 0`, `native_steps: 0`, `optimizer_updates: 0`, an exclusive `output` directory, an exact `source_sha256` filename map, and `subjects` with exactly the 22 literal roles in `admission.ROLES`. Every subject is `{path, sha256}`. `physical_arrays` supplies the ten literal roles in `admission.PHYSICAL_KEYS` with the same path/hash structure. All paths must identify the actual qualified producer inputs/outputs; old declared hashes do not permit substituted paths.

The mandatory separate clearance has `approved: true`, the exact `request_sha256`, and `review: {path, sha256}`. Its concrete review must have `passed: true` and the same `request_sha256`. CLI:

```
python source_prepared_v1/run_consistency.py --request ACTUAL_REQUEST --clearance ACTUAL_CLEARANCE --clearance-sha256 EXACT_CLEARANCE_SHA256
```

`rows.npz` retains every assembled raw `features` float32[13976,1323], `target` float64[13976,23], original `normalized_target` float32[13976,23], incoming prior/history, and origin metadata. Origins are 0=nominal, 1=physical endpoint, 2=new recovery trajectory. `origin_row` is the unchanged row within that origin. Dataset codes 0..4 name old nominal collections; physical uses 0..2; new uses 5. `origin` disambiguates repeated dataset/control identities. `phase`, `control`, `source_frame`, `plan_control`, `plan_local` and clipping flags are retained. Old plan IDs are -1 (unavailable in this minimum input set); old nominal clipping flags have `clipping_flags_known=false`, not an assertion of no clipping.

Each `aliases_MODE.npz` contains:

- `row_digest` S64[N], `group_id` int32[N] and `anchor_row` int64[N]. Nongroup rows have IDs -1.
- `members` int64[M], `offsets` int64[G+1], preserving every duplicate member without pair enumeration.
- `target_min`, `target_max` float64[G,23] and normalized-label min/max float32[G,23].
- `target_delta_from_anchor` and `normalized_label_delta_from_anchor` float64[N,23], plus raw-target RMSE/max per row. Nongroup entries are zeros and are identified by `group_id=-1`.
- `distinct_raw1323_count` and `distinct_current1000_count` int64[G].

Each `completed_MODE.json` records the mode summary immediately after its archive is saved. `proximity_queries.npy` int64[72] holds the fixed global row IDs. `proximity.jsonl` has four records per completed query (288 on success), ordered full global, current global, full same phase, current same phase. Records retain both row IDs and origin/dataset/phase/control/frame/clipping metadata, distances and target jump vector.

`report.json` records `passed=true`, `evidence_diagnosis_completed=true`, exact source/request/input/output hashes, all six summaries, exact counts and explicit zero task calls. It sets `training_selected=false`, `target_conflicts_repaired=false` and `rows_averaged_or_dropped=false`. These fields describe diagnostic execution, not a numerical consistency gate. On failure, `failure.json` preserves the exception and completed-mode/query progress; no successful report is written.
