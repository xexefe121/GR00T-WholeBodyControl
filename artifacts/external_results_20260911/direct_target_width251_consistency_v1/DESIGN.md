# Conditional saved-label consistency diagnosis

This package is source preparation only. No actual collection, task array load, checkpoint read, inference, gradient, optimizer update, native step, or training selection is performed by preparation. A completed, qualified recovery collection and a separate concrete root-selected consistency request are required before the runner can read arrays.

The diagnostic retains exactly 13,976 records: 9,904 original nominal rows, 3,054 original physical endpoint rows, and 1,018 new connected expert trajectory rows at controls 251 through 1268. The new trajectory originates in one failed-student state followed by expert execution; it is not 1,018 independently queried student states. Prefix controls 0 through 250 and terminal/hold targets are excluded.

## Input and label identity

Original nominal current features are the unchanged retained 1,000 columns of each qualified 1,069-column archive. Their saved incoming prior and history form the remaining 323 columns and must equal the existing qualified nominal context archive exactly. Physical endpoints use their saved actual inverse-applied prior and once-advanced history, checked against the qualified physical context archive. Their labels are absolute `label_fixed_map_target` values, not finite-difference physical training labels. New rows must have exact 1,000/current, 323/context, 23/prior, and 300/history concatenation identities, original global-control/frame mapping, phases 99/819/100, and committed-plan indices.

All raw 1,323 inputs remain float32. Targets remain float64 radians; the original copied `normalized_labels` function also records the unchanged float32 normalized labels. Native limits, rounded joint span, frozen mean/std and context slices must agree. No input normalization is fitted and no target is clamped, averaged, dropped, relabeled, or repaired.

## Six alias views

Each view independently hashes every row, then compares full representative bytes before accepting a digest match. An actual SHA-256 collision between unequal rows terminates the diagnostic rather than admitting an alias or performing an unbounded collision search.

1. Raw float32 1,323-column byte identity.
2. Raw 1,323-column numeric identity with signed zeros canonicalized.
3. Current 1,000-column byte identity.
4. Current 1,000-column identity with signed zeros canonicalized.
5. Float64 normalized prelayer byte identity.
6. Float64 normalized identity with signed zeros canonicalized.

The exact frozen exporter source is bound to the audited width81000 training source map. Its AST must retain `Cast(float32 -> float64)`, subtraction of the exactly promoted frozen float32 mean, then division by the exactly promoted frozen float32 std. There is no input clipping or cast back to float32 before the first layer. The equivalent source algebra is evaluated with NumPy float64; no Torch/ONNX/ORT module is imported and no kernel output is claimed. The source's later ELU `Min` and final float32 output cast do not change this prelayer definition. This replaces the earlier design note's tentative float32-normalization candidate with the actual released float64 algebra.

Duplicate groups preserve every original row index, origin, phase, target, group membership and anchor. Per-component target minima/maxima and each member's exact target difference, RMSE and maximum absolute difference from the first member are saved. Differences after the original normalized-label cast are also retained. Component ranges are not claimed to be maximum pairwise RMSE. Raw-radian disagreement and cast-level disagreement are separate summaries. Current-only aliases report how many distinct full causal inputs exist in each group. Signed-zero numeric equivalence is explicitly separate from bit identity.

## Fixed proximity work

Exactly the first 24 chronological new controls from each of three phases are queried, 72 in total: 251..274, 350..373, and 1169..1192. Every query compares with all 12,958 old nominal/physical records. Four views reuse the same distances: full/current global nearest neighbor and full/current same-phase nearest neighbor. Exact ties choose the earliest original row index.

Distances use the frozen prelayer normalization above. Full distance is the sum of squared current1000, prior23 and history300 distances. The four views report all three block RMS values, full RMS, target differences, target RMSE/max, provenance and known clipping flags. Unknown old nominal clipping flags remain explicitly unknown. No distance threshold is invented and nearest records do not select a controller, replace goals, or add source/control indices to model inputs.

## Bounded resources and preservation

The main feature array is 13,976 x 1,323 float32, approximately 74 MB. Target arrays, provenance, group membership and per-anchor residual arrays are linear in row count. Alias views execute sequentially in blocks of 256; each view performs one full row pass and at most one representative-row recomputation per row, with additional linear duplicate-member reporting. They never build a pair matrix. A large alias group's temporary byte set is bounded by the total input bytes. The exact process RSS is not certified by these array bounds.

Proximity makes exactly 72 old-candidate passes, 932,976 query/candidate comparisons and 1,234,327,248 scalar squared-distance contributions. At most 256 x 1,323 normalized candidate values are reduced per block. Four views reuse that work. No full58 endpoint corpus is loaded or expanded. Completed alias modes and every completed query are saved before subsequent work; JSON progress is flushed and fsynced. Failure preserves available artifacts and writes `failure.json`; it never triggers a retry or a completion claim.

The actual runner requires NumPy 1.26.4. Synthetic tests may use the installed Windows NumPy because they test controlled source arithmetic and schema failures; they are not a substitute for actual task-array diagnosis.

## Minimal admission

The old rows are tied to the completed width81000 fit owner, independent saved-fit audit, training request/frozen inputs and exact shared output manifest. Only consumed original archives and ten physical arrays are read. The new rows are tied to the completed collector report, actual collector request, exact collector source review and upstream recovery qualification, and the collector's declared row/normalization output hashes. There is no additional standalone collection data-review receipt and no circular requirement that this consistency diagnostic already pass.

The root-selected consistency request binds those subjects and this source map. A mandatory clearance hash binds the exact request and concrete review before the output directory is created. All consumed subjects and source files are checked again before completion. Upstream collector qualification continues to require the complete independently verified physical/source/quiet main and continuous hold; no scope or physical gate is weakened.

Successful diagnostic completion is evidence availability, not a claim that the labels are consistent or that training is selected. Conflicts and near-neighbor jumps require interpretation. No new-row student prediction errors can be obtained from this model-free calculation. Any future 1,018-row inference pass remains separately unselected.
