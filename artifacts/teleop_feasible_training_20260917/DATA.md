# Phase A data: feasible reference motions

Completed 2026-09-17. The codex run that began this phase lost its backend
connection before finishing; the results below were produced and verified
independently. See the Log in `PROGRESS.md`.

## Source clips

Public TWIST2 recordings from `https://github.com/amazon-far/TWIST2.git` at
commit `d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a`, restored from the partial
clone at `Z:\codex\twist2_inspect`. Every file was hashed as a Git blob and
compared with `git ls-tree -r HEAD`; all eleven match.

| Clip | Git blob SHA-1 |
|---|---|
| `0807_yanjie_walk_001.pkl` | `d0e6c57ef6eecca622f104ce958a68f9e775e459` |
| `0807_yanjie_walk_002.pkl` | `71597b5788a2c44b68dccf313b4860cf1020fe9e` |
| `0807_yanjie_walk_003.pkl` | `3905de55bf48eee593a521d3b561133665b5c3d5` |
| `0807_yanjie_walk_004.pkl` | `fe6319329f5b9b2e7b5bee29849c1a288912b293` |
| `0807_yanjie_walk_005.pkl` | `6c7905bac5290ab73cd2e6d4f3e2bd241ef73eeb` |
| `0807_yanjie_walk_006.pkl` | `d1dcb96b840c0e74ba4da00922ea60ecfaed9cc5` |
| `0807_yanjie_walk_007.pkl` | `ee9861e0f311bca44b87cff7482ca400a6398834` |
| `0807_yanjie_walk_008.pkl` | `a1dd65ae10dc0f4abfc44eb7cf75fea8ed78e634` |
| `0807_yanjie_walk_009.pkl` | `b8c8d8a443c5795c097c30e18f907ae6f520dc5a` |
| `0807_yanjie_walk_010.pkl` | `c7f46c806b499ae1ae8816ecbea1934dcd30e6c6` |
| `accad_A3___Swing_t2.pkl` | `c7ad3a61c797ea91ddf67502a89456e4f3337945` (restored, not imported) |

`BLOBS` in `gear_sonic/scripts/prepare_g1_true23_twist2_replay.py` was extended
with the seven additional walk hashes, each taken from `git ls-tree`. The three
original entries are unchanged.

All ten walks were imported to
`/mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walkNNN/`
(`causal_packets.json`, `motion.native23.npz`, `source_report.json`).

## Audit tool and its validation

`gear_sonic/scripts/audit_g1_true23_feasible_reference_motion.py` is a thin
adapter over existing repository functions —
`audit_g1_true23_fixed_self_contacts.measure_fixed_contacts`,
`audit_g1_true23_reference_bank_self_contacts.measure_self_contacts` and
`validated_reference_qpos`, and `g1_true23_reference_floor.reference_geometry`.
No contact logic is reimplemented. Two small corrections were needed before it
could run at all: supplying `joint_names` for motions that omit it, and passing
the floor routine the exact key set it requires. The self-contact validator's own
FK consistency check independently confirms the supplied joint order.

**The PICO capture was included as a control**, since the earlier published
audit reports its values. The adapter reproduces them exactly:

| PICO quantity | Earlier audit | This run |
|---|---:|---:|
| Fixed-body contact frames | 53 | 53 |
| Deepest fixed-body contact | 22.657 mm | 22.657 mm |
| Deepest self penetration | 114.36 mm | 114.361 mm |
| Frames needing floor lift | 5,940 | 5,940 |
| Maximum floor lift | 58.176 mm | 58.176 mm |

## Feasibility results

Fixed-body contacts are contacts between bodies outside both arm trees, which no
arm adjustment can remove. Evidence:
`feasibility_audit.json` and `feasibility_audit_sonic_library.json` under
`/mnt/e/codex-artifacts/teleop_feasible_training_20260917/`.

| Clip | Frames | Fixed-body frames | Self-collision frames (max) | Floor-overlap frames (max) |
|---|---:|---:|---|---|
| walk001 | 695 | **0** | 0 | 186 (27.6 mm) |
| walk002 | 667 | **0** | 0 | 304 (20.7 mm) |
| walk003 | 819 | **0** | 0 | 34 (20.4 mm) |
| walk004 | 637 | **0** | 0 | 220 (22.7 mm) |
| walk005 | 862 | **0** | 0 | 414 (23.6 mm) |
| walk006 | 750 | **0** | 0 | 67 (18.8 mm) |
| walk007 | 871 | **0** | 21 (21.7 mm) | 159 (18.8 mm) |
| walk008 | 364 | **0** | 0 | 42 (17.3 mm) |
| walk009 | 444 | **0** | 0 | 87 (21.9 mm) |
| walk010 | 283 | **0** | 0 | 24 (29.4 mm) |
| hand_crawl (SONIC) | 606 | **0** | 0 | 549 (**115.3 mm**) |
| elbow_crawl (SONIC) | 606 | **0** | 277 (5.1 mm) | 600 (18.1 mm) |
| happy_dance (SONIC) | 546 | **0** | 4 (0.4 mm) | 522 (13.9 mm) |
| walk001 after safe9 projection | 695 | **0** | 0 | 49 (21.7 mm) |
| **PICO capture** | 6,541 | **53 (22.7 mm)** | 1,199 (114.4 mm) | 5,940 (58.2 mm) |

The walks' uniform 17-29 mm floor overlap is a whole-body vertical offset, the
kind a common vertical translation removes. The safe9 projection already used on
existing corpus clips cuts walk001's floor-overlap frames from 186 to 49.

## Selection

Applying the pre-registered rule — zero fixed-body contact frames — verbatim:

**Training set**
- TWIST2 walks 001, 004, 005, 006, 007, 009, 010, each at weight 2.0, matching the
  weight the previous corpus gave its two walks.
- SONIC library hand_crawl, elbow_crawl, happy_dance, at weight 1.0 as before.

**Held out, never trained on**
- **walk002 — primary held-out clip.** Baseline tracking was already measured on
  it: legs12 RMSE 0.2286 rad, arms10 0.5841, all23 0.4192; pelvis-relative hand
  p95 0.4649 / 0.4493 m.
- walk003 and walk008 — secondary held-out clips, for additional evidence.

The baseline checkpoint was never trained on walks 002, 003 or 008 either, so the
comparison on those clips is between two policies that have both not seen them.

**Excluded**
- The three PICO anchor captures (upright, standing, crouch). They come from the
  PICO source the control row above shows to be geometrically infeasible.

## Known confounds, recorded before training

1. **hand_crawl sinks 115 mm into the floor** in 549 of 606 frames. It meets the
   pre-registered fixed-body rule, so it stays; the rule is not being changed
   after seeing data. If the result is ambiguous, this clip is the first
   suspect.
2. **walk007 has 21 self-collision frames** (21.7 mm). Same reasoning: it passes
   the fixed-body rule, so it stays.
3. **Two things change at once, not one.** The corpus both drops the infeasible
   PICO anchors and adds five more walks. A pass therefore supports "a feasible
   corpus helps", but cannot say which of the two changes was responsible.
