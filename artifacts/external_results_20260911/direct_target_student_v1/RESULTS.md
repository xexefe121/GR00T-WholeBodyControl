# Direct absolute-target fit

One selected fit completed all 5,000 updates from a fresh, fixed-seed initialization. The hidden process exited 0 on 2026-09-11 at 12:26:55 UTC; both processes are absent. No extra fit, checkpoint selection, BFM inference, or native physics was performed by this training stage.

The 1,000-input head removes the previous-action, previous-target, and BFM-base feature blocks. It predicts normalized absolute targets. Reconstruction uses the exact existing float32 span promoted to float64, plus the float64 default, followed by the unchanged native bounds.

Training used all 9,904 qualified nominal rows, the fixed saved velocity-pair schedule, and all 3,054 qualified physical-response pairs. Counters exactly match 70,550,000 training rows, 460,740 diagnostic Torch rows, and 601 CPU ONNX calls. GPU/CPU/ONNX maximum pre-clamp discrepancy is 0.0000067099973684 rad, below the declared 0.00001 rad threshold.

| Saved metric | Fresh initialization | Ordinary final |
|---|---:|---:|
| Equal-15-cell nominal target RMSE, rad | 0.789405 | 0.109945 |
| Nominal normalized objective | 0.0572117 | 0.00146922 |
| Full velocity-response objective | 0.0000393326 | 0.0000555893 |
| Physical-response objective | 0.000235719 | 0.000193443 |

The full velocity-response objective worsened. This tradeoff is retained; the fixed budget and ordinary final were not changed. These saved-data metrics do not establish connected stability.

Final checkpoint: `fit/student_head.pt`, SHA256 `ad6d657affba0796e7313d85ace240cd31d46e188c577da049880a7a031aecba`.

Final ONNX: `fit/student_head.onnx`, SHA256 `d35bf48cc1f755edef14dbe47a84f912ab2775c3a29a315f375d1c6d1918e088`.

Fit report: `fit/report.json`, SHA256 `1d2f6f0e8fbee8e165b731c603f241885cb8f96ab385d9753feecd682011fd59`.

Owner verification: `owner_completion_verification.json`, SHA256 `51bb731607530880177221b255e59d173646110fa16d08008c6e5b132c908fcf`.

Independent root audit: `../direct_target_root_audit_v1/results_v1/report.json`, SHA256 `82a22c814def221dc8f11a14e53ba96004308635b1c3bd76a38cb256e6ae14b8`. All 16,812 saved-result checks passed without additional model, optimizer, or physics calls.

The selected follow-on is one WSL activation witness and then the original strict 1,569-control lifecycle plus conditional continuous 250-control hold. Its source and concrete launch gates are owned by the separate evaluation artifact. Hardware remains unqualified.
