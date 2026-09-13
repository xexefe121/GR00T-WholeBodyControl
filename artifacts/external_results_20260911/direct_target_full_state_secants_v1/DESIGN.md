Source preparation only. No generation, fit, model call or native rollout is selected by this package.

Generate signed finite changes along all 58 coordinates of each of the 3,057 existing committed teacher maps. Preserve the 140,622 old joint-velocity rows as exact overlap; add 213,990 rows for the other 35 axes. Full output is 354,612 signed rows, plus the unchanged 3,057 nominal centers. This entails 357,669 pure feature/map evaluations, zero BFM/head/native calls, and no replanning.

| Teacher tangent indices | Group | Fixed signed radius |
| --- | --- | --- |
| 0–2 | World root position | ±0.001 m |
| 3–5 | Root rotation in saved plan quaternion chart | ±0.01 rad |
| 6–28 | Native joint position | ±0.01 rad |
| 29–31 | Root linear velocity in original state coordinates | ±0.05 m/s |
| 32–34 | Root angular velocity in original state coordinates | ±0.25 rad/s |
| 35–57 | Native joint velocity | Existing ±0.01 × native per-joint cap |

These are bounded local probes, not coverage of every recorded departure. At the first actual departure, control 251, root-position change peaks at 0.000433 m, root rotation at 0.005994 rad, root linear velocity at 0.03332 m/s, and root angular velocity at 0.57169 rad/s. The proposed angular-rate radius is comparable to the old joint-rate radii, 0.20–0.37 rad/s, but does not cover that entire first root-rate vector. At 252, other state directions already dominate the saved map's preclip feedback change.

All saved nominal joint positions have strict native margin greater than 0.01 rad: observed minimum 0.015556881582736104 rad over 3,057 × 23 entries. Consequently no adaptive radius, clipping, shrinking or row deletion is needed. The generator rechecks these exact strict margins and old velocity margins before generation. Each perturbed state must also pass exact native joint-position bounds, strict native joint-speed bounds, positive root height, finite values and quaternion norm tolerance 1e-10. These static checks do not establish contact consistency or dynamical feasibility. They do not change the native controller oracle.

The rotation probe is formed in the existing teacher's plan chart: compute the original `difference(plan, center)`, change only its chosen rotation-vector coordinate, compose the plan quaternion with the original quaternion exponential, and normalize the resulting quaternion as the original integration helper does. It does not compose a center-local rotation and pretend that is a pure plan-axis change. A principal-chart boundary is an explicit failed slot/attempt. The recomputed full 58D difference must change only the requested axis within 2e-12 roundoff. Linear coordinates mutate the original center arrays directly; this preserves old joint-velocity rounding exactly.

Use the byte-preserved pure `DirectFeatures` implementation. Every probe rebuilds measured state and all future goals from its new position, quaternion and velocity at the unchanged source frame. In particular, yaw changes rotate goal positions, rotations and velocities using the new measured heading. History, prior actions, BFM base targets and BFM goal latents are not feature inputs. No forward kinematics, latent inference or physics is needed.

Before creating probe outputs, verify all 3,057 unperturbed 1,000-feature vectors against the existing retained feature indices `[0:52] + [75:1023]`, and verify their full original `K @ difference` targets with both feedback ±0.1 and native target clips. Then generate and verify all 140,622 old velocity rows before any added-axis row. Compare reduced features, full target, raw feedback, both clip masks and exact perturbed joint velocity bytes. All added rows retain the original full gain product and clips. Existing zero-gain and saturated maps remain present.

Save per-row qpos/qvel, full tangent and tangent change, signed physical radius, raw feedback/correction/preclip/target, raw float64 target change, clip masks, 1,000 float32 features and committed status. Save axis group/unit metadata and each center's dataset, control, source frame, phase and one of nine cells. Float32 features must differ from their center and opposite sign; an alias is a preserved explicit failure, never divided by an arbitrarily small displacement. Cross-corpus duplicate/conflict review is still required before any future fit; this preparation does not authorize fitting.

No new loss weight or perturbation denominator is chosen by generation. Existing native target spans remain exactly the original float32 values promoted to float64 when used. Raw float64 target changes and physical radii allow a future learner to choose finite-change versus secant normalization explicitly. For a future finite-change mean, the full-axis requested denominator per cell would be `center_count × 58 × 2 × 23`; an equal-six-group mean would instead retain each group's `center_count × group_axes × 2 × 23` denominator before averaging six group means. Cell center counts are `[100,819,100]` for each of three datasets. No masks redistribute weight or remove nominal centers. A failure leaves its requested slot uncommitted, preserves reason and active returned arrays, and prevents a complete dataset receipt.

The generator draft requires a future explicit selected request with all consumed source/input hashes and pinned NumPy/SciPy versions. The draft creates no request or launcher. Progress records distinguish the old-overlap phase and added-axis phase; all writes precede status commit. Final input/source/request rehash is mandatory. Existing artifacts remain unchanged.
