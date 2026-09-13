# Why the next change targets feedback sensitivity

The step-55,000 policy has lower nominal action error but fails earlier in closed-loop native physics. On the saved velocity probes, its normalized finite-secant error is 0.680371, compared with 0.334851 for a policy with zero response to those perturbations. The discrepancy remains on unclipped examples. This is evidence about the saved local teacher maps, not a new expert rollout.

The independent 42-state decomposition also shows incomplete directional coverage: existing explicit velocity probes cover 23 of the teacher's 58 tangent coordinates. The other 35 coordinates contribute materially before the first target clamp. At control252, the saved root-angular-velocity feedback-change contribution has RMS0.17623rad, compared with0.07949rad from all joint velocities. These norms are not causal percentages; terms can cancel and the committed map becomes stale away from its recorded trajectory.

Matching derivatives as well as values is an established learning objective. [Sobolev Training](https://arxiv.org/abs/1706.04859) studies this approach for function approximation and policy distillation. It supports the method family, without establishing stability for this humanoid.

[TaSIL](https://arxiv.org/abs/2205.14812) connects derivative-aware imitation to trajectory error under assumptions on expert incremental input-to-state stability. Those assumptions are not certified for this clipped, contact-rich, committed-plan teacher. The paper therefore motivates measuring feedback fidelity; its guarantees cannot be transferred to this experiment.

[Tube-guided MPC augmentation](https://arxiv.org/abs/2306.00286) uses robust tube MPC structure to generate robust imitation data. No certified tube or invariant set exists in the current implementation. Applying saved local gains to new states must remain labeled finite-map augmentation, rather than a newly replanned or certified robust expert.

Selected next preparation: explicit signed probes in all58 state directions, fixed physical radii, exact original23-direction overlap, unchanged target clips, full nominal anchors, and per-direction/group diagnostics. The separate fixed-batch gradient measurement informs loss scaling; it cannot establish final weights or closed-loop performance by itself. Actual training and physical acceptance remain separate tests.
