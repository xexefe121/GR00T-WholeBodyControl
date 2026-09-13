# Split first-layer arithmetic review

The proposed correction is mathematically sound. Keep the single 256-by-1323 parameter tensor, use contiguous slices for both normalized inputs and weights, and calculate:

```
old = F.linear(z[:, :1000].contiguous(), w[:, :1000].contiguous(), bias)
context = F.linear(z[:, 1000:].contiguous(), w[:, 1000:].contiguous(), None)
first_activation = ELU(old + context)
```

The existing second and third layers, feature normalization, checkpoint parameter names, fixed losses, two conditions, initialization, schedule, learning rates and budgets remain unchanged. Contiguous copies must remain connected to autograd; detaching or recreating parameters would invalidate the proposal. Bias belongs in the old-input branch only.

Independent synthetic tests passed on CPU and CUDA. Initial first-layer outputs match the 1000-input reference byte for byte for diagnostic batch sizes 1, 52, 176, 238 and 256, and CUDA training batch sizes 1728, 3054 and 9904. Complete CPU MLP outputs also match. Analytic gradient checks confirm both weight slices receive gradients, the bias receives one contribution, and normalized-zero blinded context produces zero appended-column gradients. With nonzero context weights, split and monolithic float64 algebra agree within 1e-12 on a synthetic fixture. These tests load no task data or task weights and perform no optimizer, ORT or native calls.

This is not a universal bitwise identity proof. Adding positive zero can change negative-zero sign bits. Kernel choice, normalization and real saved inputs still require the declared complete-corpus initial check. Preserve the existing 1e-5 rad initial drift limit, failed original outputs and all attempted/returned diagnostic accounting. Record split execution in restoration/report metadata rather than leaving the old changed-MatMul description unqualified.

The monolithic internal-float64 export is the same real-number function of the six saved tensors. Its rounding differs from split float32 training, so the existing measured export checks remain mandatory. This correction grants no export or controller qualification by itself.

For the next saved audit, require the same original-65000 parameter and normalization identities, zero appended weights at initialization, fresh empty AdamW, exact restored RNG, per-condition initial saved predictions, shared paired initialization, blinded final appended weights/moments remaining zero, and all unchanged final parity/manifest/owner checks. The first failed attempt consumed 1437 producer GPU32 forwards covering 367570 rows and zero updates; saved auditors themselves made zero task calls. Keep these scopes distinct.

No completed-pair audit or corrected task-model execution was performed by this review. Final implementation source review and actual root-selected request remain separate gates.
