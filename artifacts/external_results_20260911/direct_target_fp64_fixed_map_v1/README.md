The saved FP64 rollout departs before native target clipping. Control 250 starts with exact qualified features but 0.045540 rad target RMSE; at 251 and 252 the errors are 0.093250 and 0.305625 rad. First learned clipping is 256. Correct applied-target history and all release/runtime bindings passed the separate 5,283-check semantic audit.

This selected array-only diagnostic evaluated all 42 actual learned states against their matching-clock, already committed query250 plans. All 42 nominal target reconstructions were byte-exact under pinned NumPy 1.26.4. Original full 23-by-58 gain multiplication and both feedback ±0.1/native target clips are unchanged. No inference, physics, optimization or replanning ran.

| Preclip feedback-change RMS, rad | Other 35 axes | Joint velocity 23 axes |
| --- | ---: | ---: |
| Control 251 | 0.09043 | 0.09576 |
| Control 252 | 0.20153 | 0.07949 |
| Before first learned clip, 251–255 | 0.22730 | 0.14211 |
| All departed states, 251–291 | 1.60581 | 0.76955 |

Other 35 axes exceed the joint-velocity contribution in 37 of 41 departed states, including 3 of 5 before clipping. At 252, root angular velocity alone contributes 0.17623 rad RMS. Joint-velocity-only supervision therefore omits material directions of the recorded feedback departure. Existing physical one-step labels do include some coupled state changes; this result does not claim those directions are wholly absent from training.

At 251, actual joint-velocity departure is 0.41752 rad/s RMS, or 0.013225 RMS and 0.032652 maximum after division by native per-joint speed limits. At 252 those normalized values are 0.034129 and 0.131329. The fixed ±1% velocity probes are local axis probes, not coverage of these full coupled vectors.

Head-versus-map RMSE grows from 0.090819 rad at 251 to 0.292396 at 252 and 1.17054 at 291. The maps' own feedback clip activates in 41 rows and their native target clip in 2 rows. They are stale committed maps, not replanned expert truth or safety certificates.

Group contributions are calculated both from the actual plan-relative tangent and from actual-minus-nominal tangents in the same plan chart. This avoids substituting a different quaternion chart for the change. Six group arrays and the original full products are retained. Grouped summation changes floating accumulation order and is checked against a roundoff bound; grouped values never produce commands. Group norms are not additive attribution percentages: cancellation and nonlinear clipping matter.

Evidence: `report.json` SHA256 `e78567ac2e1fa1eb2c5f4b1ab388a2bca7838aa21a75ae0970d8eb485806b43b`; request `6a726a5222c53bc79c57e9640d092ffd016f59662fc7014095b8a2d18f150508`; source `6d5c927bb88effdc4f4347f09c158c79e80383bae0fe6b7050960a52796bdab6`. Four synthetic tests cover all 58 axis memberships, cancellation, original nested clips and same-chart subtraction. The earlier semantic report is `../direct_target_fp64_saved_semantics_review_v1/report.json`, SHA256 `64f0b4382d21109420bfbcc2dace856655ac8e4a9d0fbff1d48b39792a784506`.
