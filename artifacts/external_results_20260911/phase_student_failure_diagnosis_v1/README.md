The final65000 controller failed during acquisition, at control278/substep4, after the exact 250-control BFM standing prefix and 28 complete learned controls. Right hip yaw reached +32.141383524 rad/s against its 32 rad/s native limit. Source motion was not reached and the separate hold was not run. The ordinary final checkpoint and failure trace remain preserved.

All 279 recorded feature vectors, BFM sensor states, measured-history vectors, prior actions, combined actions, and applied targets reconstruct exactly from the saved trajectory. The final five named history arrays also match exactly. This rules out a transition indexing or history-buffer implementation discrepancy in the tested runtime. The first enabled input at250 matches the qualified expert label exactly; its actual ONNX output agrees with the saved Torch prediction within2.384e-7 rad.

| Control | Actual target error versus matching expert (rad RMS) | Saved fit error at matching expert input | Input distance to matching expert | New event |
|---|---:|---:|---:|---|
|250|0.031958|0.031958|0|Fitting error on an exactly covered input; no clipping|
|251|0.150848|0.017650|0.057857|First actual input divergence; no clipping|
|252|0.331045|0.029981|0.150853|First native target clipping: left knee,0.033534 rad|
|253|0.592371|0.022112|0.272122|First training-envelope excursion, in previous left-knee action|
|278|1.406743|0.037737|1.637294|Right hip yaw speed failure|

Input distances use explicit float64 RMS differences standardized by the unchanged original mean/std. The minimum distance to any of the3,057 moving-phase training inputs grows from0 at250 to1.565849 at278. These distances have no fitted acceptance threshold. The coordinate-wise training envelope and matching-input divergence are separate diagnostics. After250, matching expert labels and saved predictions belong to a different expert trajectory; they are not expert counterfactual commands at the later actual student states.

At251 the BFM base changes by0.006698 rad RMS from the matching expert base, while the head output changes by0.149672 rad RMS. Initial fitting error and subsequent amplification therefore precede clipping. All named measured-history fields first differ from the matching expert at252, because the actual sensor/action trajectory has already changed and the four-sample history correctly exposes that change with its declared delay.

There is also a later action-history semantic mismatch between the tested student and MPC supervision. The student stores the combined action before native target clipping. The expert stores the normalized actually applied native target. At252 those values first differ materially, by0.095630 normalized action units for the clipped left knee. This difference becomes the explicit prior-action input at253 and appears in the history-actions vector at254. By255 the maximum difference is3.949574. This is not an issue with clipping actions to±5: the correct expert semantics can legitimately exceed±5. It is the distinction between an unclipped requested target and the native target actually applied.

The failing hip-yaw target oscillates from+1.074010 at276 to-1.714436 at277 and+2.384556 rad at278. At278 the BFM base is+0.232438 and learned residual+2.152118 rad. The four applied actuator torques are88.0,84.666509,52.167696,30.254753 Nm; velocities are7.880626,19.958919,27.656857,32.141384 rad/s. Commands and actual forces remain within native effort limits. Full2ms state/command arrays are saved in `arrays.npz` for independent physical replay.

Most direct bounded next experiment: use the same frozen65000 head and change only residual-phase previous-action recording to the normalized actually applied native target, matching the expert labels; retain original raw BFM action semantics during standing and terminal modes. Require the canonical trajectory to remain exact through the first clipped command at252, then assess one continuous lifecycle under the same gates. This needs separate authorization and source review. It removes an identified post-clipping semantic mismatch without a new fitting budget or expert query, but it cannot remove the demonstrated initial fitting error or pre-clipping amplification. Further gain/architecture decisions should use the separate analytical head-sensitivity review, not infer that this small correction guarantees stability.

This diagnosis made zero model-inference calls, optimizer updates, or physics steps. No next candidate was launched.
