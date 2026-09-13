# Ordinary step 65,000 full-state finite-feedback fit

The selected 10,000 updates completed once in 531.0 seconds. Actor and normalization started exactly at 55,000; AdamW started with empty moments and ended at 10,000. All fixed row/call budgets passed. No controller, BFM or native physics ran in this stage.

The one initial three-loss gradient calibration set the full-state coefficient to 1.8188207859141674. Its Euclidean combined direction had positive initial descent dots for all three losses; this is neither an AdamW-step nor closed-loop stability guarantee. No weight sweep or later recalibration occurred.

| Saved GPU32 objective | Initial | Final | Change |
| --- | ---: | ---: | ---: |
| Nominal 15-cell MSE | 0.000448774960629 | 0.000446065195138 | -0.6038% |
| Full-state 54-cell response MSE | 0.000369290543407 | 0.000219506030042 | -40.5601% |
| Physical 9-cell response MSE | 9.56462217552e-05 | 9.45231297139e-05 | -1.1742% |
| Fixed weighted objective | 0.00121609449878 | 0.000939830454926 | -22.7173% |

Comparisons use the same complete-corpus GPU32 diagnostic definitions. The full-state term matches bounded committed-feedback finite changes across declared physical neighborhoods, without dividing by physical radii. It does not represent replanned expert behavior or a common-unit Jacobian error.

Final CPU64/GPU64/ORT64 public-float32 export parity passed at maximum 2.9388367295268836e-8 rad, against the unchanged 1e-5 preclamp gate. The ordinary float32 learned parameters are promoted exactly for 64-bit normalization and layer arithmetic; no float32 ONNX release was attempted.

Owner verification passed: durable raw/final exits 0, both wrapper 10408 and child 23164 absent, 216 input + 15 source pins and all output-manifest files unchanged. Independent saved-fit review and canonical physical qualification remain separate gates; this training result does not qualify closed-loop performance.

- Fit report: fit/report.json —7426d281be79c3610ef02ae8889c3ef66264da15a9667551d6bca4166a4beaea
- Checkpoint: fit/student_head.pt —8a1b67e09285a77910d62dd1b2214c8d3684004e55a82041544e750006b29a0a
- ONNX: fit/student_head.onnx —045f04138610a06a0171a899d002cfa4f03e30e316dc9442199c833c16e43902
- Owner: owner_completion_verification.json —8bacc308af12b0c888313848a8b20fdce04e11e252cb62cc850a3102efb5fce5
