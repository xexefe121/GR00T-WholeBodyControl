# Fixed group-energy response balancing

This preparation defines one loss change. It does not select training, an endpoint, or controller execution.

For group g, fix E_g to the audited teacher zero-response normalized MSE, averaged equally across the nine dataset/phase cells. Set w_g = mean(E_0,...,E_5) / E_g. The six weights, in the original tangent-group order, are:

| Group | Fixed weight |
|---|---:|
| root position | 5.901096772528486 |
| root rotation | 0.7122640541039265 |
| joint position | 5.684180872160203 |
| root linear velocity | 0.44567938947852637 |
| root angular velocity | 0.5454780461142876 |
| joint velocity | 5.759716089786164 |

`balanced_full_state_loss` calls the byte-preserved original loss to compute its 54 cells, then multiplies by this six-weight sequence repeated nine times. Every original endpoint-minus-center calculation remains live in autograd; float32 predictions are still widened before subtraction against the original float64 target change divided by the existing native span. No radius divisor, target modification, clipping modification, sampling change, or per-cell removal is introduced. The original metric and all original cells are returned alongside the weighted objective and weighted cells.

The mean weighted zero-response energy equals the original mean energy, within float64 rounding. This keeps the original zero-response scale while raising the relative importance of the three lower-energy feedback groups. It does not equalize gradient norms or establish a descent direction for nominal or physical losses. Weights depend only on teacher energies; fitted predictions and the actual rollout failure do not choose their values.

Candidate engineering continuation, subject to separate root selection and concrete source review:

- Restore the exact ordinary68000 causal endpoint, all six AdamW states and RNG, the unchanged shared1323 normalization, and the split float32 first layer. Retain actual prior23 plus incoming history300, including the already-verified applied physical prior.
- Use N + 1.8188207859141674 * F_balanced + P. Keep all9904 nominal anchors,354612 full-state signed rows and3054 physical endpoints, existing denominators, labels and center/context timing. No new coefficient calibration.
- Propose a fixed3000-update continuation to ordinary71000, with optimizer counters3000→6000. Reuse the same saved3000 schedule rows once, without new draws. Proposed cosine learning rate1e-6→1e-7 continues from the old endpoint's rate without an upward restart. Weight decay1e-5 and clipping10 remain unchanged. These optimizer/budget choices are proposals, not authorization.
- Save the ordinary final endpoint only. Retain original and balanced full-state metrics/cells separately throughout; continue to report original zero-response and group-relative errors. Do not replace the original metric with the weighted one in comparisons.
- Preserve all initial saved-output/restoration checks and full final five-backend diagnostics. Keep the monolithic float64 export with float32 public input/output, the existing1e-5 rad numerical gate, a separately selected one-call WSL witness, and original1569+conditional250 canonical acceptance. No threshold relaxation or extra rollout is part of this preparation.

At the proposed3000 updates, training would remain9000 model forwards/44,058,000 rows. Initial/final GPU32 plus final CPU64/GPU64 diagnostics would total5748 Torch forwards/1,470,280 rows; final ORT would add1437 calls/367570 rows. These are prospective counts; this package has performed none of them.

A single balanced continuation can be evaluated as an engineering attempt. It cannot establish that balancing caused improvement relative to merely receiving more optimization. That causal comparison would require a separately selected continuation control with the same optimizer, rate and schedule. No such control, fit or controller trial is selected here.
