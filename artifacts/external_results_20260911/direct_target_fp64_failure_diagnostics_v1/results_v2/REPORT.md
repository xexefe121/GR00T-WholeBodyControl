# Saved FP64 failure diagnostic

The original 1569-control lifecycle failed at control 291/substep 6, 5.832 s. Right hip roll reached -22.398976 rad/s against 20 rad/s. Root independently reproduced all 2916 native steps. Zero of 819 source controls and zero hold controls executed.

Both learner runs share the expert's complete initial state and exact 250-control startup. New target first differs at control 250; initial 23-joint target RMSE is 0.045539955 rad. First position divergence appears at physics step 2501; first different precontrol is 251.

New first learned target clamp: control 256, versus earlier direct5000 control 270. CSV files preserve every learned control's target, right hip-roll state, clamp count and maximum joint-speed ratio.

New 42 head-only-phase proposal times: median 0.559200 ms, p95 0.910905 ms, maximum 3.589000 ms. Startup 250 BFM timings are separately reported. The timer includes measured-state/history staging, features, head call and output checks; it excludes commit/native stepping and is not a pure ORT timer or independent 500 Hz plant proof.

Earlier direct5000 failed at 6.316 s on left ankle pitch range. Longer survival is not a tracking qualification; both stop before source begins at 7 s. Same-clock expert targets are descriptive comparisons, not counterfactual teacher labels at learner states. Training and export both differ between the two heads, so this comparison does not isolate precision as a cause.

All source/input hashes are pinned and rechecked. This artifact uses saved arrays only: zero model, physics, optimizer or teacher-map calls. See report.json for checks and full limitations; failure_comparison.png/.svg for the plot.
