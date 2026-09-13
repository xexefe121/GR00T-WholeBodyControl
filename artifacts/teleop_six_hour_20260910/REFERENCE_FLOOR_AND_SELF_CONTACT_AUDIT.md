# Selected v3 reference geometry

All 10,674 frames of the four selected portable multistart references were audited against the exact compiled MuJoCo 3.2.3 native23 collision model. The selected files are in `E:\codex-artifacts\sonic23_teleop_six_hour_20260910\mjbatch_intent_inputs_v1`. The older `intent_retarget_v3` directory is a preserved superseded single-seed experiment and is not the selected reference.

The floor is native geom 0, a horizontal plane at world z=0. Native foot collision geoms are spheres 15–18 and 30–33, each with radius 5 mm. Their analytic signed floor clearance exactly matches actual MuJoCo contact distances in every observed foot-floor contact. No collision mask, exclusion, floor, root force or physical model parameter was changed.

| Selected recording | Frames | Maximum minimum upward lift | Raw lift speed maximum | Raw lift acceleration maximum |
|---|---:|---:|---:|---:|
| PICO | 6541 | 58.176 mm | 0.4632 m/s | 23.458 m/s² |
| walk002 | 1428 | 20.675 mm | 0.2823 m/s | 14.049 m/s² |
| walk003 | 1580 | 20.371 mm | 0.2920 m/s | 14.599 m/s² |
| walk008 | 1125 | 17.272 mm | 0.1868 m/s | 6.924 m/s² |

PICO needs upward correction in 5,940 frames. Its minimum lift median is 32.20 mm and p95 is 46.82 mm. During its first three source seconds, the worst left/right foot surface penetrations are 47.93/42.53 mm. The maximum lift required to clear any body geometry is the same as the foot requirement, so common upward translation sufficient for the feet also clears other floor contacts in these references.

Both feet and all other collision geometry are more than 2 mm clear in 587 PICO frames. Those are geometric clearance frames, not proven physical flight: the source has no contact labels, and none of these PICO frames has a finite-difference COM vertical acceleration within 2 m/s² of ballistic gravity. The minimum upward correction is zero on these frames; no downward anchoring or artificial contact is introduced.

A raw `max(0, -minimum_foot_clearance)` correction has substantial derivative peaks. A causal smoothed profile must retain an explicit floor guard or use declared preview; blindly smoothing the minimum lift can reintroduce floor penetration.

The separate v4 candidate has now passed independent full-frame geometry and transform verification. Its fixed critically damped filter has omega 15/s, a 20 mm buffer and a 1 micrometre floor guard. All 10,674 frames clear the native floor; minimum clearances are PICO 1.640 mm, walk002 9.493 mm, walk003 7.824 mm and walk008 8.415 mm. Native joints, joint velocities, orientations, angular velocities and frame rate are bit-exact to selected v3. All-body common-Z translation error is at most 1.11e-16 m, native FK error 4.44e-16 m and added linear-velocity derivative error 5.66e-15.

An independently implemented matrix-exponential filter reconstruction agrees with exported lift within 5.55e-17 m. Changing every future raw lift after five selected cutoffs per clip changes no earlier pose output. All four clips use identical parameters, and the floor guard never activates. PICO's added vertical speed/acceleration maxima are 0.12447 m/s and 1.89298 m/s². The filter's pose output is causal; the exported central-difference linear velocity explicitly uses one future pose, or 20 ms. Its 20 mm positive buffer also raises already-clear standing poses; PICO's initial foot clearance increases from 3.01 to 23.01 mm. These checks establish geometric floor compatibility, not dynamic tracking or support contact.

Independent v4 evidence is in `E:\codex_sonic_runtime\reference_floor_audit_20260910\v4_pico`, `v4_walks` and `v4_filter_independent_verification.json`. Self-collision counts and worst depths remain unchanged, as expected from common rigid translation.

Floor correction cannot remove self collision. Actual native `mj_forward` contact constraints, with `exclude=0`, report PICO torso/left shoulder yaw penetration 114.36 mm at frame 3842 (source 69.62 s), and left hip roll/hand penetration 105.15 mm at frame 1532 (23.42 s). These pairs have contype/conaffinity 1/1 and active constraint addresses. The largest walk002/walk003/walk008 self penetrations are hand/hip-roll pairs at 32.73/26.04/49.27 mm.

Bounded local arm projection clears the actual self contacts at PICO frames 1532, 4991 and 2565, retaining strict original relative hand/head error below 0.15 m. Their maximum task errors are 0.11048, 0.08803 and 0.08001 m, and both adjacent reference steps fit within 80% of native speed limits. Root, waist, legs, head and source clock remain unchanged. These are reference geometry probes, not physical controller validation.

The deepest frame 3842 cannot be repaired by the tested one-step constrained solve. A static refinement clears actual contacts and reaches relative hand errors 0.14721/0.14867 m, but needs a 0.91747 rad arm change and has no speed qualification. The optimizer hit its iteration limit; geometry and task checks were evaluated independently rather than inferring success from solver status. A temporal repair would need to start earlier in the collision episode, which spans frames 3756–3864. No full collision-aware retarget has been produced.

Contact-point analytic gradients were unreliable for deeply overlapping mesh signed distances, so the bounded feasibility solve retains analytic hand Jacobians and uses centered finite differences for collision-distance constraints. Every candidate is checked again using actual native collision generation, including newly created pairs.

The full fixed-body audit found 53 PICO frames with contacts that arm motion cannot change: 48 exceed 1 mm and 35 exceed 5 mm. The deepest is a knee/knee contact of 22.657 mm at frame 1464. Other pairs are left/right hip yaw (6.728 mm maximum), pelvis/right hip roll (15.365 mm), torso/right hip roll (5.991 mm) and torso/right hip yaw (6.777 mm). The three walking references have no such fixed-body contacts. These findings are preserved in `v4_fixed_body_contact_audit.json`; arm-only feasibility never implies an entirely self-collision-free reference.

A two-frame projection with declared 20 ms pose preview accepted the first 14 frames of a bounded PICO episode, then stopped at frame 3768 because its proposed next hand error was 0.150592 m. A separate warm-start refinement with 180 iterations reaches maximum hand error 0.148900006 m at frames 3768/3769, with no arm contacts. Its incoming/outgoing joint steps use 5.38%/24.67% of the allowed 80%-native speed envelope. Fixed pelvis/hip penetrations of 9.56/10.67 mm remain. The optimizer still reaches its iteration limit; the actual geometry/task/speed checks pass independently. This is a local boundary refinement, not a completed temporal reference or a physical rollout. Both the failed attempt and refinement remain in `pico_collision_episode_armonly_v3` and `pico_collision_boundary_refinement_v1`.

The authorized bounded continuation stops again at frame 3840: its proposed frame 3841 has right-hand error 0.150974 m. Independent native `mj_forward` verification of the 86 accepted frames, 3754–3839, finds no arm contacts, no joint-bound excess and hand/head maxima 0.148966/0.149665/0.025104 m. Actual adjacent arm steps use at most 68.21% of the permitted 80%-native speed envelope. Nevertheless, the accepted fragment reaches 20.19 rad/s and 1192.38 rad/s² arm motion; no physical tracking was demonstrated. Returning directly to the next original source pose exceeds the speed envelope by a factor of 1.668. Fixed-body contacts persist up to 15.365 mm. This incomplete fragment cannot be substituted into a full reference. The continuation, independent check and failures remain under `pico_collision_episode_armonly_resume_v1` and `pico_collision_episode_independent_fragment_audit_v2.json`; no full v5 reference was exported.

Full arrays and receipts: `E:\codex_sonic_runtime\reference_floor_audit_20260910\selected_v3`. Worst-frame evidence and strict independent checks: `pico_selfcollision_probe_v2`; tightened static feasibility: `pico_worst_static_refinement_v2`. The superseded preliminary audit is preserved under `v3` with its original reference paths and hashes.

## Original-native canonical derivative pair

The later A/B ablation uses original native poses, independently of the multistart retarget above. A preserves all original positions, quaternions, joints, frame rate and timeline exactly and replaces velocity channels with declared central derivatives. B adds a freshly computed common-Z floor correction to A, using the same omega 15/s and 20 mm buffer. The packs are `mjbatch_original_canonical_inputs_v1` and `mjbatch_original_canonical_floor_inputs_v1` under the E: task archive.

Independent checks passed all 15,938 variant-frames across PICO and walk002. A poses match both the preserved original archive and native portable bundle bit-for-bit. Joint and body linear derivatives match direct central differences exactly; separately implemented quaternion multiplication/log agrees with world angular derivatives within 9.4e-15 rad/s. Full-frame native FK position/quaternion errors stay below 5.6e-16. Both references retain native joint and adjacent speed bounds. Central derivatives use one future pose, explicitly 20 ms.

B's common saved Z translation agrees within 1.11e-16 m. Independently computed raw floor lift matches exactly and a matrix-exponential reconstruction agrees with the applied filter within 5.56e-17 m. No guard activates. Its minimum foot clearances are PICO 1.649 mm and walk002 9.553 mm; no nonfoot floor contacts remain. Joint positions/velocities and all body orientations/angular velocities remain exact between A and B. PICO still has 498 self-collision frames with maximum penetration 83.320 mm; walk002 has no self-collisions. Geometry clearance and derivative validity do not establish dynamic support or tracking.

Evidence: `E:\codex_sonic_runtime\reference_floor_audit_20260910\original_canonical_pair_verification.json`, `original_canonical_A/report.json`, and `original_canonical_B/report.json`.

The original-pose PICO reference also has 61 frames of fixed-body contact, with 19.335 mm maximum knee/knee overlap at frame 1464, 2.404 mm hip-yaw/hip-yaw overlap, and 7.421 mm pelvis/right-hip overlap. Walk002 has no fixed-body contact. Evidence: `original_canonical_fixed_body_contacts.json`.

## Frozen-state goal sensitivity

The independent review of `bfm_goal_counterfactual_v1` passed all saved-array, hash, clocking and pair-metric checks. Each clip uses the same 100 measured pre-control states, histories and previous actions for all five goal variants. Original-goal action and target outputs reproduce the saved baseline exactly. Source frame is control + 11; previous action is control - 1. Recomputed cosine and raw/clipped target RMS metrics agree exactly with the producer report.

Original-to-A isolates velocity convention changes; A-to-B isolates floor lift and its matching Z velocity. A-to-v3 changes poses and their matching derivatives together. The latter produces latent cosine medians 0.6383/0.3697 and arm-target RMS differences 0.2350/0.3127 rad for PICO/walk002, substantially larger than A-to-B's 0.0110/0.00358 rad arm changes. These are policy sensitivity measurements at original-trajectory states, not evidence of closed-loop quality or training-distribution support. Raw reference-feature statistics are computed before outer goal feedback. Evidence: `bfm_goal_counterfactual_independent_verification.json`.
