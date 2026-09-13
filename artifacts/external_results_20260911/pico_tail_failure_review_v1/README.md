# PICO stop at control3800: saved-evidence review

Read-only analysis; no controller, inference, solver or physics run. Exact source trace SHA-256: `419ceb38d95ae1a6314089b56274cea2e576153b72bfa78e6d2f276886299612`. Full hashes and reproduction script in `report.json` / `analyze_saved.py`.

All 20 executed plans in controls3700–3799 retained feasible nominal H30 incumbents; all 100 imminent full-native ten-step certificates passed. Failed request3800 executed no rejected controls. All 13 shifted-seed rejections among 21 requests first arise at knots25–29, after the retained25 controls. This establishes safe retained prefixes under each saved nominal rollout, not viability beyond them or full-native H30 equivalence.

Local tracking deteriorates before stopping. Between source67–68s and68–69s, unchanged-original world-root error p95 rises .06279→.15367m, original heading p95 8.85→16.54°. Declared-v4 per-control leg RMS p95 rises .16993→.32451rad. Declared left ankle body-origin world error p95 rises .05681→.16145m. These feet/joint diagnostics are explicitly against v4; they are not the original-relative-foot qualification gate. A transient leg excursion appears at3712, recovers, then deterioration builds from3740 and becomes sharp from3780. Pointwise original heading first exceeds15° at3796 within this reviewed window. Root error stays below .20m pointwise here.

Replay mismatch or feedback clipping is not supported as the cause: no feedback component clips in these100 controls. Largest actual-versus-planned precontrol joint difference is .00265rad; root difference is30.2µm. Six committed plans3750–3775 have zero gains, but later plans have nonzero gains. Last-plan max gain26475 does not imply large applied feedback: saved raw correction remains near zero in the final second.

The final shifted tail exactly equals reference feedforward at frames3836–3840. Its boundary target jump is RMS .2911rad / max .8051rad; retained-prefix adjacent jumps are larger (RMS median .4598rad / p95 2.2578rad, maximum component5.6814rad). Command jumps are not actual joint-speed violations. Therefore neither tail placement nor discontinuity alone establishes causation.

The separate exact-state3800 hold-tail probe also rejects: full-native first failure at physics step258 =516ms, right ankle pitch bound. Its frozen report is bound in `report.json`. Thus a simple hold substitution is not a supported fix. Earlier3770 tail tests use a different actual trajectory and do not settle this state.

One evidence-backed next test, already authorized by parent: refine the saved guided restoration final proposal using K=0 rollout feedback, retaining the original shifted-seed regularization anchor and unchanged tracking objective, horizon, iteration budget, limits and tolerances. This uses guided progress without discarding it at the fallback boundary. Only the ordinary final proposal may be admitted, after both nominal H30 and exact full-native certificates. This review does not claim that test succeeds, establishes recursive feasibility, or closes the offline-to-live timing gap.

Later result: this one authorized refinement completed and still rejects. Merit7.27871→7.07455; no actual controls executed. Exact report/certificate bound in `later_refinement_result.json`. The suggested experiment is exhausted; no controller fix is established by it.
