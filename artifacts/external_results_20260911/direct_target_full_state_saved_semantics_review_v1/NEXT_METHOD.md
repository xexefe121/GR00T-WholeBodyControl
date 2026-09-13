The next structural candidate should be a wider state/goal network, with the
existing 1000 inputs, supervision and target arithmetic retained. This is a
capacity hypothesis, not a conclusion that width caused the failure. No new fit
is selected by this note.

The completed65000 audit verifies all297 issued controls and47 learned inputs,
head outputs, history, applied-target feedback, release identities and native
clock. The original250-control BFM prefix and first activation witness match.
The right knee exceeds its20rad/s limit at control296/substep9; the independent
replay reproduces all2969 native samples. Source motion and hold never begin.

At the exact activation input, target error is0.04676rad RMS. At251 the input has
departed, target error is0.10617rad versus the nominal expert and0.09709rad versus
the matching frozen map. At252 those errors are0.25542 and0.24446rad. Native target
clipping begins only at257. Thus changing clipped feedback or repairing history
bookkeeping cannot explain this initial amplification; the checked bookkeeping
is already correct and the head does not consume it.

The current finite-response fit remains1.1381times worse than zero response on
the balanced54-cell objective. Every aggregate tangent group remains worse than
that baseline, and97.995percent of its error is antisymmetric between signs.
This establishes poor local directional fitting, including within the supervised
neighborhoods. It does not identify whether the bottleneck is width, optimization
or omitted conditioning. The nominal objective barely improves in this stage,
so simply increasing response weight risks trading away another necessary part
of the task.

The actual departure is coupled: the other35 tangent axes contribute more than
joint velocity in44 of46 departed rows and5 of6 pre-clipping rows. At251 joint
velocity reaches3.17percent of a native speed cap, versus the isolated probe
radius of1percent; root angular-velocity departure is0.375rad/s RMS, versus each
axis radius0.25. Better axis fitting therefore remains necessary evidence to
seek, but cannot itself certify recovery of this already larger coupled state.
The fixed maps remain stale same-clock maps, not replanned expert commands.

Concretely, prepare1000→512 ELU→512 ELU→23. Embed the old256-unit blocks and biases
unchanged. Initialize new units separately, with zero connections from them into
the old path and zero final outgoing weights, so the mathematical initial output
is the65000 function. Avoid identical duplicated units that remain symmetric
under deterministic training. Require an actual initial numerical gate before
updates: changed matrix dimensions can change floating-point accumulation even
when the real-valued function is identical.

Keep the same9904 nominal,354612 full-state and3054 physical rows, their cell
weights, the fixed1.8188207859 response coefficient, and the native span/clamp
contract. Avoid simultaneously adding causal history or changing loss weights;
otherwise improvement cannot be attributed to the structural candidate. The
optimizer/LR initialization and fixed update budget still require an explicit
choice: the previous run's fresh-AdamW reset is an optimization confound, not a
reason to silently copy or fabricate optimizer state for enlarged tensors.

Assess the ordinary final checkpoint on both nominal accuracy and the already
defined per-group response-versus-zero baseline before any new canonical trial.
Those metrics are diagnostic prerequisites to judging the proposal, not a proof
of stability or a substitute for the unchanged strict1569+conditional250 test.

Causal previous-command conditioning remains a subsequent hypothesis. The
previous-target hold has21.95times the nominal MSE of the direct head, which
rejects a naive skip/hold remedy. It does not test whether prior/history could
disambiguate near-identical observations. Current evidence cannot justify
claiming either that causal conditioning is needed or that it is useless.
