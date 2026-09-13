# Current observer feedback, not a stronger guard

The after-inference momentum correction passes the former25.58s virtual range
failure but reaches an unverified native ankle target at32.60s. On the identical
prefix leg RMSE improves0.124231→0.119676rad and root p950.228086→0.218382m, still
not acceptable tracking. A coordinated guard can find a next-step feasible
target but costs500ms and needs>2rad target change; do not adopt it in this trial.

New causal timing hypothesis: updated internal missing velocities must enter
the same current policy observation as current retained physical measurements.
The previous version queried the policy before applying momentum corrections,
which reached3.97rad/s. Move measurement assimilation BEFORE policy inference;
replace only the latest explicitly virtual q/dq history row. Older history and
real23 measurements remain untouched. An explicit copied-measurement hook in
a separately versioned benchmark avoids edits to any executed shared source.

Require tests for exact old/new benchmark physics with a no-op history hook,
copy isolation, observer update ordering, no double assimilation, matching
measurement identity, history alignment and failed-state behavior. Reverify
all four full source-only prediction traces before ONE full native PICO trial.
Source truth for missing axes is never passed to the observer. Matched source
predictions must remain within1e-5, missing range excess<=1e-6.

Use the same complete115.60s source and standing lifecycle, original normal
encoder/decoder,920ms received buffer, native23 plant/gains/efforts, existing
source target conversion and OLD inward-only range preview. No coordinated
guard, extra damping, higher cap, widened bound, virtual reset or source rewrite.
Preserve failure and compare both equal-prefix and full available results.
Do not turn simulated survival into tracking/deployment readiness. No hardware.
