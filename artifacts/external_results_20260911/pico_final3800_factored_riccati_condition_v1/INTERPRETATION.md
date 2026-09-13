# Final3800 numerical diagnosis

These are private counterfactual diagnostics from the stopped canonical PICO run. No actual controls executed, no source frames advanced, and no shared controller source changed.

The complete run stopped with `no_feasible_seed` after3800 lifecycle controls/3450 source controls/38,000 physics steps. Every actual step was valid, but the full6530-control lifecycle is incomplete. Rejection is not successful recovery.

One fixed hold-tail proposal retained the first25 shifted commands and repeated command24 for the last five. It failed right ankle pitch at516ms, so the new reference tail is not the only issue.

One zero-feedback restoration refinement started from the saved guided proposal, with regularization still anchored to the original shifted targets. It reduced merit7.278706→7.074547 but failed left ankle roll at282ms; all ten generated sequences were infeasible. The backward recursion stopped after numerical positive-definiteness failures exhausted the existing regularization cap.

An algebraically factored Riccati value update reduced cancellation substantially on the archived matrices. A separately authorized single solve using only that algebraic change lowered merit to6.682088, but its ordinary final proposal still failed: left ankle roll at282ms, right knee maximum excess0.00263167rad, and maximum speed ratio1.086264. All28 generated sequences were infeasible. This update has not been integrated into the shared controller.

The final factored proposal's backward recursion fails at knot13. Its regularized control Hessian has scale1.499e17, while the original control curvature is0.0002. Binary64 epsilon times matrix norm is33.29. State-space regularization adds `mu * B.T @ B`; with mu1e6 and B norm3.872e5, this inflates the matrix scale about24,145-fold from the unregularized6.209e12.

A matrix-only standard control-space LM alternative, `Qreg = R + B.T @ V @ B + I`, retains scale6.209e12 and has numerical minimum eigenvalue near0.999. It has not been used in a solve. Exact arithmetic gives `lambda_min(Qreg) >= lambda_min(R) + lambda` when V is positive semidefinite. The stored V has a tiny negative computed eigenvalue(-5.2e-10 against maximum1.401e7); multiplying a worst-case bound by B norm squared is too loose to prove the alternative always works in floating point. The existing positive-definiteness rejection must remain, and SPD alone does not establish good conditioning or safe resulting controls.

The next proposed bounded experiment is one explicitly declared control-space damping variant with unchanged physical predicates, merit, source, alpha schedule and maximum iterations. No new solve or full restart has been launched after the matrix assessment.

All original inputs, source snapshots, derivative arrays and failing matrices are retained in this directory and the sibling refinement directories. `control_lm_assessment.json` records the native WSL NumPy version; matrix eigenvalues at these extreme scales can differ across BLAS/LAPACK builds. This uncertainty is itself evidence of ill conditioning, not a relaxed acceptance threshold.
