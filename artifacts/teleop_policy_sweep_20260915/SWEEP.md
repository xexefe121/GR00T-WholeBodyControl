# Paired checkpoint sweep on one recorded PICO clip (2026-09-15)

Every paired encoder/decoder checkpoint in `artifacts/g1_true23_frozen_lora/`
was run against the **same** recorded clip
(`original_sonic_happy.true23.causal_packets.json`, SHA-256 `237910ad...dc9d`),
the same plant, the same gains and the same gates, via
`record_g1_true23_saved_teleop_diagnostic.py`. Simulator only; no transport,
headset, DDS, robot channel or policy change.

This is a balance/survival comparison. It is **not** a tracking qualification —
that metric is not produced by this runner.

| Checkpoint | Passed | Controls | Min base height (m) | Max base tilt (rad) | Fallback |
|---|---|---:|---:|---:|---|
| `ieee_motion_ppo_20260906_v2/model_100` | true | 535/535 | 0.6341 | 0.1777 | none |
| `ieee_motion_ppo_resume300_20260906_v1/model_200` | true | 535/535 | 0.6380 | 0.1805 | none |
| `ieee_motion_ppo_resume300_20260906_v1/model_300` | true | 535/535 | 0.6395 | 0.1881 | none |
| `native_support_stateful_20260905_v1/breadth50/model_50` | true | 535/535 | 0.6311 | 0.2237 | none |
| `native_support_training_20260905_v1/breadth200/model_50` | true | 535/535 | 0.6405 | 0.2103 | none |
| `paired_encoder_20260905_v2/original_breadth25/model_25` | true | 535/535 | 0.6311 | 0.2245 | none |
| `projection_cost_20260906_v1/baseline100/model_100` | true | 535/535 | 0.6317 | 0.1955 | none |
| `projection_cost_20260906_v1/penalty2_100/model_100` | true | 535/535 | 0.6331 | 0.2011 | none |
| `reset_curriculum_20260906_v1/model_100` | true | 535/535 | 0.6359 | 0.1721 | none |
| `reset_curriculum_20260906_v1/model_400` | true | 535/535 | 0.6352 | 0.3509 | none |
| `standing_motion_ppo_20260906_v1/model_100` | true | 535/535 | 0.6430 | 0.1962 | none |
| `standing_retention_ppo_20260906_v1/model_100` | true | 535/535 | 0.6250 | 0.2309 | none |

`stage_one_training_20260905_v1/breadth50/model_50` exited 1; see its log.

## What this shows

**All twelve survive the clip identically**: 535 of 535 controls, fallback never
engaged, minimum base height clustered in 0.625–0.643 m and maximum tilt in
0.17–0.35 rad. Survival does not discriminate between these checkpoints at all.

Two consequences:

1. A passing balance result on this clip carries almost no information about
   checkpoint quality. Any of these twelve would produce the "teleop runs"
   headline. Selecting on that basis would be selecting on noise.
2. The discriminating measurement has to be **tracking fidelity** — how closely
   the robot follows the operator's reference motion — which
   `record_g1_true23_saved_teleop_diagnostic.py` does not compute, and which
   every one of these reports declares as `tracking_fidelity_qualified: false`.

The measurement that does discriminate is
`measure_g1_true23_saved_teleop_tracking.py`, but it accepts only a
`g1_true23_public_twist2_replay_import_v1` source report — i.e. one of the three
pinned public TWIST2 walk clips — not this dance bundle. Producing that import
requires the pinned source recordings `0807_yanjie_walk_{002,003,008}.pkl`.

For scale, the previously recorded tracking numbers for the paired breadth25
pair on those clips (PROGRESS.md, 2026-09-08) were heading p95 about 14–15
degrees and pelvis-relative hand p95 about 0.16–0.19 m, against a positive
control at 9.58 degrees and 0.055 m. That gap, not balance, is the problem.

---

# Tracking fidelity ranking (2026-09-15)

Balance did not discriminate, so the twelve surviving checkpoints were scored on
the quantities the packets themselves carry: the native23 joint reference
`q_ref23_native` and the pelvis-local three-point VR targets
`vr_3point_local_target`. Script: `rank_fidelity.py` (copy in this directory).

This is **not** the pinned `measure_g1_true23_saved_teleop_tracking`
qualification, which accepts only a public TWIST2 import report and needs the
pinned clips `0807_yanjie_walk_{002,003,008}.pkl`; only `walk_001` is on this
machine. It reuses the same reference quantities, the same physical model and
the same body offsets so candidates can be ranked against one another.

## Two corrections made while building this metric

Both were caught by sanity-checking rather than by any error message, and both
had produced confident-looking but meaningless numbers:

1. **Joint order.** `q_ref23_native` is stored in NATIVE_IL23 order, which pairs
   left/right adjacently. MuJoCo `qpos` is in hardware order, which groups whole
   legs. Differencing them directly gave a uniform ~0.73 rad leg RMSE for every
   checkpoint — the giveaway being that unrelated checkpoints scored identically.
   The repository's own `native_to_hardware_compact()` is the correct conversion.
   After it, the initial state matches packet 0 to 0.032 rad maximum.
2. **Off-by-one.** 536 recorded states pair with 535 packets as state `k` to
   packet `k`, so the final state has no packet and is not scored.

## Result

| Checkpoint | Leg RMSE (rad) | All-joint RMSE (rad) | L hand p95 (m) | R hand p95 (m) |
|---|---:|---:|---:|---:|
| `projection_cost_20260906_v1/baseline100/model_100` | 0.2680 | 0.3413 | **0.1738** | **0.1677** |
| `native_support_training_20260905_v1/breadth200/model_50` | 0.2715 | 0.3413 | 0.1793 | 0.1693 |
| `paired_encoder_20260905_v2/original_breadth25/model_25` | 0.2716 | 0.3435 | 0.1801 | 0.1697 |
| `native_support_stateful_20260905_v1/breadth50/model_50` | 0.2744 | 0.3440 | 0.1802 | 0.1697 |
| `projection_cost_20260906_v1/penalty2_100/model_100` | 0.2705 | 0.3373 | 0.1817 | 0.1722 |
| `ieee_motion_ppo_resume300_20260906_v1/model_200` | 0.2684 | 0.3066 | 0.1994 | 0.1806 |
| `reset_curriculum_20260906_v1/model_400` | 0.2667 | 0.3074 | 0.2009 | 0.1784 |
| `ieee_motion_ppo_20260906_v2/model_100` | 0.2696 | 0.3018 | 0.2010 | 0.1860 |
| `reset_curriculum_20260906_v1/model_100` | 0.2712 | 0.3060 | 0.2038 | 0.1817 |
| `ieee_motion_ppo_resume300_20260906_v1/model_300` | 0.2696 | 0.3069 | 0.2045 | 0.1815 |
| `standing_retention_ppo_20260906_v1/model_100` | 0.2807 | 0.3069 | 0.2097 | 0.1909 |
| `standing_motion_ppo_20260906_v1/model_100` | 0.2740 | 0.3049 | 0.2100 | 0.1856 |

**Cross-check against numbers this session did not produce:** leg RMSE lands at
0.267–0.281 rad against the 0.292 rad `MATCHED_WALK003.md` reports for frozen
SONIC, and hand p95 at 0.174–0.210 m against the 0.16–0.19 m PROGRESS.md
recorded for the breadth25 pair on the TWIST2 clips. The metric agrees with
independently recorded results.

## What the ranking says

**Selecting a better existing checkpoint does not fix this.** The best candidate,
`projection_cost_20260906_v1/baseline100/model_100`, reaches 0.1738 m hand p95.
The prepared positive control in `MATCHED_WALK003.md` reaches 0.055 m on its
matched task. The entire field spans 0.174–0.210 m — a 21 percent spread, all
sitting roughly **three times worse** than demonstrated feasibility. They are
twelve samples from one failure regime, not a range containing a good answer.

The modest reordering is still worth noting: `projection_cost` baseline leads on
hand error while `ieee_motion_ppo`/`reset_curriculum` variants lead on all-joint
RMSE, so the objectives are trading off against each other rather than one
dominating.

## A structural limit specific to 23-DoF, not a policy defect

The head p95 is **0.0000 m for every checkpoint**. That is not perfect tracking.
Across the whole clip the head reference varies by a standard deviation of
1.07e-07 m — it is constant to within numerical noise, while the hands vary by
0.13–0.23 m.

The cause is kinematic. The three-point head target is the torso body offset by
(0, 0, 0.35), expressed in the pelvis frame. A 23-DoF G1 has exactly one waist
joint, `waist_yaw`, a rotation about z. A z-offset point is invariant under a
z-rotation, so the head point cannot move relative to the pelvis **no matter
what the policy does**.

Consequence for full-body teleop on 23-DoF hardware: of the three VR tracking
points, only the two hands carry recoverable information. Operator head motion
relative to the pelvis is unrepresentable on this embodiment — it is not
something a better checkpoint or more training can recover. Any acceptance
criterion that counts head tracking as a passing screen on 23-DoF is measuring
a constant.
