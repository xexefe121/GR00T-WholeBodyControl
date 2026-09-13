All3,057 moving-phase expert commands reconstruct bit-for-bit from the saved planned target, planned state, local gain, actual state, feedback correction clip, and native target clip. No inference, dynamics, optimizer work, derivative-label dataset, or augmented target dataset was generated.

The original hybrid uses the exact preterminal prefix of the original full-MPC producer. That producer stores `planned_state`, `planned_target`, `feedback_gain`, and both raw/applied corrections per control. The query1 and query250 branches instead store immutable complete plans; the audit maps each executed control to its committed plan and local index. Original producer and qualified hybrid qpos/qvel/targets agree exactly through control1268. The audit binds all consumed archives and source snapshots in `input_hashes.json`.

|Moving cohort,1,019 controls each|Velocity gain spectral norm median / p95 / max|Feedback-clipped controls|Native-target-clipped controls|Controls with velocity derivative clipping kinks|Zero-gain controls|
|---|---:|---:|---:|---:|---:|
|Original|0.914 /1.588 /11.548|5|23|54|0|
|Query1|0.915 /1.598 /489.861|0|11|35|5|
|Query250|0.923 /1.718 /559.824|0|12|21|45|

Gains have units rad/(rad/s), or seconds. Maximal velocity gains occur at query1 control712 and query250 control1097. Their largest entries are168.534 and304.991. The corresponding coordinate-axis interval preserving all current interior clipping branches can be extremely small: approximately9.3e-15 rad/s at query1 control712 and0.000328 rad/s at query250 control1097. The former is effectively on a numerical clipping boundary, despite exceeding the audit's absolute1e-12 target-boundary test. These are not robust derivative witnesses over useful finite perturbations. Query250 control1178 also has spectral norm64.700 and a smallest interior axis radius approximately1.68e-5 rad/s.

Most diagonal velocity gains are negative, but substantial positive cross-phase diagonals are present. Counts positive/negative/zero are3,693/18,186/1,558 for original;3,606/18,082/1,749 for query1;3,250/17,550/2,637 for query250. Do not reverse signs or force every diagonal to be damping. At query250 controls250/251, velocity spectral norms are0.9403/0.7287, with no clipping kink. The45 zero-gain query250 controls and five query1 controls follow plans with no accepted optimizer update. They represent a constant frozen committed target, not proof that a freshly replanned MPC optimum has zero state sensitivity.

The joint-position block is much less suitable for unqualified global regularization: spectral maxima are95,289,86,602, and66,179. Most actual state deviations from saved plans are tiny, so enormous gains can coexist with negligible recorded corrections and a successful nominal trajectory. Exact command reconstruction does not establish reliable behavior far from that trajectory.

The native58 tangent is:

- Root position0:3.
- Right-multiplicative body orientation3:6.
- Joint positions6:29.
- Raw velocity difference29:58: root translation29:32, root angular32:35, joint velocities35:58.

`difference(planned, actual)` uses actual-minus-planned for Euclidean coordinates. Orientation uses `log(conjugate(q_planned)*q_actual)`; the corresponding integration is `q_planned*exp(delta)`. The backward solve already supplies the negative sign in K. The committed control map is

`u(x) = native_clip(u_plan + clip(K*difference(x_plan,x), -0.1, +0.1))`.

Holding the plan fixed, its joint-velocity derivative at an actual expert state is exactly `J_T = M_native M_feedback K[:,35:58]` on smooth clipping branches. Strictly clipped output rows have zero local derivative; clipping boundaries can have different one-sided derivatives and cannot be assigned a unique ordinary Jacobian. The audit identifies kink rows and reports the known smooth rows separately. Its1e-12 boundary tolerance is a numerical audit convention, not a proof of a useful neighborhood. Root orientation needs the derivative of the quaternion-log difference at the actual orientation, not simply the orientation columns of K when planned and actual orientations differ.

This is the exact derivative of the **frozen committed feedback map**, evaluated in real arithmetic from the saved numerical gains. It is not an exact derivative of fresh MPC replanning, seed selection, active contacts, optimizer acceptance, warm-target shifting, or a different future plan. Finite-difference/contact noise remains embedded in those saved gains. The network also does not receive the complete plan, gain, warm start, or commitment index.

The feature chain for an instantaneous joint-velocity perturbation holds qpos, prior action, measured history, and reference fixed. The direct1069 input change is identity at columns23:46. The BFM actor's52-state input also changes at23:46. Its goal latent remains fixed because its current-frame transformation uses qpos, not current joint velocity. Current measured terms are pushed after extracting the actor's four preceding history samples, so the actor history remains fixed for this same-call derivative. The changed BFM base additionally enters feature columns1023:1046 and the additive output base itself. Explicit prior-action columns1046:1069 and clipped prior-target columns52:75 stay fixed.

Other direct proprio mappings are joint position to0:23; free angular velocity to46:49; projected gravity to49:52; prior target to52:75; root linear velocity in heading coordinates to75:78; and root height to78. In the BFM52-state vector, angular velocity enters49:52 scaled by0.25 and gravity enters46:49. Joint-position changes also affect the BFM base. Root pose changes require differentiating heading-relative goals and the BFM goal latent; treating arbitrary1069 feature coordinates as independent physical perturbations would be wrong.

Let `b(v)` be the BFM base target, `phi(v,b)` the raw1069 features, and `r_theta(phi)` the residual already scaled into radians. For joint velocity, define `B_v = db/dv`, `S_v` as insertion into feature23:46, and `E_b` as insertion into1023:1046. On a smooth student target branch:

`J_student = M_student [B_v + J_r (S_v + E_b B_v)]`.

Here `J_r = diag(native_span) J_actor diag(1/original_std)` if differentiating the normalized actor. Copying K into the residual network is wrong: the necessary total derivative includes the BFM base and its feature path. Equivalently, residual changes along physical velocity directions would need `J_T - B_v`, while still using the complete feature chain. Saved plans alone do not supply B_v. This audit did not measure it.

A minimal possible future experiment is a soft joint-velocity derivative regularizer at the existing qualified expert states, in addition to the unchanged pointwise target/residual loss. In dimensionless form, compare `D_span^-1 (J_student-J_T) D_velocity` within each existing dataset/phase cell. Declare the velocity scale, coefficient, smooth-branch/trust-region rules, and treatment of zero-update plans before fitting. Match total applied-target derivatives, not residual partial derivatives. Preserve pointwise data unchanged. This would be an explicit frozen-feedback regularizer, not newly qualified demonstrations or a guarantee of closed-loop stability.

The present evidence does **not** support blanket K supervision across all rows: near-boundary neighborhoods, extreme contact-sensitive gains, zero-update plans, hidden planner state, and the missing BFM derivative are unresolved. Augmented local target labels would need equally explicit small perturbation bounds, unchanged history/prior, recomputed BFM base/features, and the full two-stage clip. They must be named frozen-feedback pseudo-labels; no physical feasibility or fresh-replanning credit follows from arithmetic alone. No numerical coefficient, new fit budget, or label-generation run is selected here.
