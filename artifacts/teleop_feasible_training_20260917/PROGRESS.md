# Feasible-reference training and sim test (started 2026-09-17)

## Hypothesis being tested

Earlier frozen-LoRA training used a corpus that included the PICO captures shown
to be geometrically infeasible for a 23-DoF G1 (498 self-collision frames, 53
frames of fixed-body leg/torso contacts). All 34 full replays of the resulting
checkpoints failed on target intersection. Clean public TWIST2 walks carry no
fixed-body contacts.

This experiment changes exactly one variable: **the training corpus is restricted
to references with zero fixed-body contacts.** Trainer, task, network, gains and
limits are unchanged, so any difference is attributable to the data.

## Pre-registered decision rules

Set before any training starts, so the result cannot be argued into success.

- **Comparison baseline:** `projection_cost_20260906_v1/baseline100/model_100`,
  the best hand-error checkpoint from the 2026-09-15 sweep.
- **Held-out clip:** one feasible TWIST2 walk excluded from training.
- **Pass:** on the held-out clip, completes all controls without fallback, AND
  beats the baseline on both pelvis-relative hand p95 and all-joint RMSE.
- **Kill:** if the training smoke fails, or if the trained checkpoint does not
  beat the baseline on the held-out clip, stop. Do not extend, re-tune or re-run
  the recipe. A failure here means reference feasibility was not the limiting
  factor, and further PPO on this recipe is not worth running.

## Phases

- [x] A. Data: restore all TWIST2 clips, inventory existing internet PICO data,
      audit feasibility with existing tooling, select feasible set
- [x] B. Training: corpus from feasible clips only, smoke, then bounded main run
- [x] C. Sim test: export pair, live ZMQ teleop loop, pinned tracking
      measurement on held-out clip, compare with baseline

## Log

### 2026-09-17 — Phase A: codex run interrupted, work salvaged and verified

The codex phase A run **did not complete**. Its connection to the ChatGPT
backend failed (`stream disconnected before completion`, five of five reconnects
failed) and the process exited with status 0 without writing `DATA.md`. An exit
status of 0 is therefore not evidence of completion here; everything it did
leave behind was checked independently rather than trusted.

Verified as correct:

- **All eleven TWIST2 clips restored**, each confirmed against
  `git ls-tree -r HEAD` of `https://github.com/amazon-far/TWIST2.git` at
  `d5c7108`: `0807_yanjie_walk_001` through `_010` and `accad_A3___Swing_t2`.
  All eleven match.
- **`BLOBS` in `prepare_g1_true23_twist2_replay.py` extended** with walks 001,
  004, 005, 006, 007, 009 and 010. Every added hash equals the `git ls-tree`
  value; the three original entries (002, 003, 008) are unchanged.
- **All ten walks imported** to
  `/mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walkNNN/`
  (`causal_packets.json`, `motion.native23.npz`, `source_report.json`).

Checked and accepted with a note:

- Codex added `gear_sonic/scripts/audit_g1_true23_feasible_reference_motion.py`
  (untracked). The brief prohibited a new audit framework. On reading it, it is a
  thin adapter: all contact and floor logic comes from existing functions —
  `audit_g1_true23_fixed_self_contacts.measure_fixed_contacts`,
  `audit_g1_true23_reference_bank_self_contacts.measure_self_contacts` and
  `validated_reference_qpos`, and `g1_true23_reference_floor.reference_geometry`.
  Nothing is reimplemented.

Not done by codex:

- The feasibility audit never produced output. It was started separately with
  the PICO capture included as a **control**: the earlier audit found 53
  fixed-contact frames for PICO, so the adapter must reproduce roughly that figure
  before its verdicts on the walks are trusted.

A note on verification itself: the first clip-hash check in this salvage
reported all eleven clips as mismatched. That check was wrong, not the clips —
Windows Python passed a POSIX-style `/z/...` path to git, which returned nothing
to compare against. Rerun with `Z:/...`, all eleven match. Recorded because it
would have been easy to act on the false alarm.

### 2026-09-17 — Phase A complete

Audit adapter validated against the PICO control, reproducing the published
audit exactly (53 fixed-contact frames, 22.657 mm). All ten TWIST2 walks and all
three SONIC library clips have zero fixed-body contacts. Training set: walks 001,
004, 005, 006, 007, 009, 010 plus the three SONIC clips. Held out: walk002
(primary), walk003 and walk008. Three confounds recorded in `DATA.md` before
training begins. Full detail: `DATA.md`.

### 2026-09-17 — Phase B1: FAILED

**Smoke FAILED (not started):** corpus builder stopped before loading any source because required manifest was absent at `/mnt/e/codex-artifacts/teleop_feasible_training_20260917/corpus/manifest.json`. Per pre-registered kill rule, corpus build was not rerun and the two-iteration training smoke was not started.

Seven required safe9 projections completed first. They used reference projection mode `canonical_upright_ankle`; walks 009 and 010 also used `--minimum-frames 510`. The projection report kind was `g1_true23_pico_sonic_safe_image_projection_v1` for every walk. Each motion payload array is `float32`; only `fps` scalar is `float64`. No conversion needed: corpus builder loads six motion arrays with `np.asarray(..., dtype=np.float32)` before concatenation.

| Walk | Changed frames | Max joint change (rad) | Final frames | Dtype |
|---|---:|---:|---:|---|
| `001` | 542 | 0.1603838982749042 | 695 | motion arrays `float32`; `fps` `float64` |
| `004` | 524 | 0.22404685114486345 | 637 | motion arrays `float32`; `fps` `float64` |
| `005` | 666 | 0.16038389853668217 | 862 | motion arrays `float32`; `fps` `float64` |
| `006` | 427 | 0.16038389853668214 | 750 | motion arrays `float32`; `fps` `float64` |
| `007` | 485 | 0.16038389853668214 | 871 | motion arrays `float32`; `fps` `float64` |
| `009` | 83 | 0.16038389853668214 | 510 | motion arrays `float32`; `fps` `float64` |
| `010` | 127 | 0.16038389853668217 | 510 | motion arrays `float32`; `fps` `float64` |

Commands run (Linux commands were placed in a file, copied to `/root`, CRLF-normalized with `sed -i "s/\\r$//"`, then executed):

```bash
mkdir -p /mnt/e/codex-artifacts/teleop_feasible_training_20260917/safe9 /mnt/e/codex-artifacts/teleop_feasible_training_20260917/corpus
/root/venvs/teleop23/bin/python /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic/scripts/project_g1_true23_pico_library_motion.py --repository-root /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof --input /mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walk001/motion.native23.npz --output-npz /mnt/e/codex-artifacts/teleop_feasible_training_20260917/safe9/walk001.canonical.safe9.npz --output-json /mnt/e/codex-artifacts/teleop_feasible_training_20260917/safe9/walk001.canonical.safe9.json --reachable-raw-abs 9.0 --root-mode canonical_upright_ankle
/root/venvs/teleop23/bin/python /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic/scripts/project_g1_true23_pico_library_motion.py --repository-root /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof --input /mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walk004/motion.native23.npz --output-npz /mnt/e/codex-artifacts/teleop_feasible_training_20260917/safe9/walk004.canonical.safe9.npz --output-json /mnt/e/codex-artifacts/teleop_feasible_training_20260917/safe9/walk004.canonical.safe9.json --reachable-raw-abs 9.0 --root-mode canonical_upright_ankle
/root/venvs/teleop23/bin/python /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic/scripts/project_g1_true23_pico_library_motion.py --repository-root /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof --input /mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walk005/motion.native23.npz --output-npz /mnt/e/codex-artifacts/teleop_feasible_training_20260917/safe9/walk005.canonical.safe9.npz --output-json /mnt/e/codex-artifacts/teleop_feasible_training_20260917/safe9/walk005.canonical.safe9.json --reachable-raw-abs 9.0 --root-mode canonical_upright_ankle
/root/venvs/teleop23/bin/python /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic/scripts/project_g1_true23_pico_library_motion.py --repository-root /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof --input /mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walk006/motion.native23.npz --output-npz /mnt/e/codex-artifacts/teleop_feasible_training_20260917/safe9/walk006.canonical.safe9.npz --output-json /mnt/e/codex-artifacts/teleop_feasible_training_20260917/safe9/walk006.canonical.safe9.json --reachable-raw-abs 9.0 --root-mode canonical_upright_ankle
/root/venvs/teleop23/bin/python /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic/scripts/project_g1_true23_pico_library_motion.py --repository-root /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof --input /mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walk007/motion.native23.npz --output-npz /mnt/e/codex-artifacts/teleop_feasible_training_20260917/safe9/walk007.canonical.safe9.npz --output-json /mnt/e/codex-artifacts/teleop_feasible_training_20260917/safe9/walk007.canonical.safe9.json --reachable-raw-abs 9.0 --root-mode canonical_upright_ankle
/root/venvs/teleop23/bin/python /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic/scripts/project_g1_true23_pico_library_motion.py --repository-root /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof --input /mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walk009/motion.native23.npz --output-npz /mnt/e/codex-artifacts/teleop_feasible_training_20260917/safe9/walk009.canonical.safe9.npz --output-json /mnt/e/codex-artifacts/teleop_feasible_training_20260917/safe9/walk009.canonical.safe9.json --reachable-raw-abs 9.0 --root-mode canonical_upright_ankle --minimum-frames 510
/root/venvs/teleop23/bin/python /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic/scripts/project_g1_true23_pico_library_motion.py --repository-root /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof --input /mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walk010/motion.native23.npz --output-npz /mnt/e/codex-artifacts/teleop_feasible_training_20260917/safe9/walk010.canonical.safe9.npz --output-json /mnt/e/codex-artifacts/teleop_feasible_training_20260917/safe9/walk010.canonical.safe9.json --reachable-raw-abs 9.0 --root-mode canonical_upright_ankle --minimum-frames 510
/root/venvs/teleop23/bin/python /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic/scripts/build_g1_true23_pico_fullbody_corpus.py --repository-root /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof --manifest /mnt/e/codex-artifacts/teleop_feasible_training_20260917/corpus/manifest.json --output /mnt/e/codex-artifacts/teleop_feasible_training_20260917/corpus/corpus.npz --catalog /mnt/e/codex-artifacts/teleop_feasible_training_20260917/corpus/corpus.catalog.json --spans /mnt/e/codex-artifacts/teleop_feasible_training_20260917/corpus/corpus.spans.json --recovery-metadata /mnt/e/codex-artifacts/teleop_feasible_training_20260917/corpus/corpus.recovery.json --episode-frames 500
/root/venvs/teleop23/bin/python /root/phase_b1_inspect.py
```

Corpus result: **0 sources loaded; total frames unavailable**, because builder failed resolving manifest before `_load_entries` ran. Intended manifest had seven projected TWIST2 clips (weight 2.0) and three fixed SONIC clips (weight 1.0), but it was not present at specified E: output path. Builder accepts absolute motion paths; it did not reject requested output location or require copies into Z:.

Last output lines from failed corpus build:

```text
File "/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic/scripts/build_g1_true23_pico_fullbody_corpus.py", line 62, in build_fullbody_corpus
  manifest = manifest_path.resolve(strict=True)
File "/root/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/lib/python3.10/pathlib.py", line 1077, in resolve
  s = self._accessor.realpath(self, strict=strict)
File "/root/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/lib/python3.10/posixpath.py", line 437, in _joinrealpath
  st = os.lstat(newpath)
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/e/codex-artifacts/teleop_feasible_training_20260917/corpus/manifest.json'
```

Smoke result: **not run**. There is no `smoke_run.log`, because required corpus was not created and phase stops at this failure.

### 2026-09-17 — Phase B1: corpus built, smoke PASSED

Codex completed the seven safe9 projections, then stopped correctly at a failure:
the manifest had not been written, so the corpus builder raised
`FileNotFoundError` on it. Codex did not work around the failure, as instructed.
The manifest was then written and the rest of B1 run directly.

**Safe9 projections**, all verified: float32 like the existing corpus clips,
report kind `g1_true23_pico_sonic_safe_image_projection_v1`.

| Walk | Output frames | Changed frames | Max joint change (rad) |
|---|---:|---:|---:|
| 001 | 695 | 542 | 0.1604 |
| 004 | 637 | 524 | 0.2240 |
| 005 | 862 | 666 | 0.1604 |
| 006 | 750 | 427 | 0.1604 |
| 007 | 871 | 485 | 0.1604 |
| 009 | 510 (padded from 444) | 83 | 0.1604 |
| 010 | 510 (padded from 283) | 127 | 0.1604 |

Cross-check: the existing corpus's own walk001 safe9 clip reports 541 changed
frames and 0.16038 rad maximum change, against 542 and 0.1604 here. Walks 009 and
010 were padded to 510 frames, matching the existing `hold510` convention.

**Corpus:** 10 sources, 6,593 total frames (existing corpus: 8 sources, 6,035),
at `/mnt/e/codex-artifacts/teleop_feasible_training_20260917/corpus/`.

**Smoke:** 2 iterations, exit 0, 2.15 s per iteration, checkpoints
`frozen_lora_model_0.pt` and `frozen_lora_model_2.pt` written to `E:`. The trainer
accepts a run directory outside the repository.

**Recipe identity check.** `resolved_training.json` was diffed against the
baseline's. Seven keys differ, all accounted for:

- `agent.max_iterations`, `agent.save_interval`, `planned_updates`: the smoke's 2
  against 100. The main run uses 100.
- `frozen_platform_lora.span_sidecar.path` and `.sha256`,
  `recovery.curriculum_metadata_sha256`: the new corpus, as intended.
- `stage_one_actuation.source_sha256`: `bc4dab24…` against `1ffce4af…`. This file
  is `gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json`, which was
  converted from CRLF to LF on 2026-09-15 to satisfy its approval hash. Those are
  exactly its CRLF and LF hashes recorded at the time. The conversion replaced
  line endings only, so the parsed actuation configuration is identical. It is
  nonetheless a genuine hash difference and is recorded as such.

No training hyperparameter, gain, limit or network setting differs.

**Main run launched** under systemd unit `feasible-train`: 100 iterations,
output `/mnt/e/codex-artifacts/teleop_feasible_training_20260917/main_run/`.

### 2026-09-17 — Phases B and C complete. VERDICT: KILL

The first main-run launch under `systemd-run` was stopped 41 s in when the WSL
distro idled out; it was rerun in an attached foreground session and completed
100 iterations in about five minutes. Export succeeded.

On held-out walk002 the new checkpoint beat the baseline on all23 RMSE
(0.4164 against 0.4192) and right-hand p95 (0.4366 against 0.4493), but was worse
on left-hand p95 (0.4721 against 0.4649). The pre-registered rule required both
hands, so the verdict is **KILL**. Secondary clips show only sub-3-percent
differences in mixed directions.

The training scalars explain it: both the baseline and the new run average about
four control steps per episode, terminated by `stage_one_actuation_guard`, with
reward near −99 throughout. The recipe barely moves the policy from its warm
start. Per the pre-registered rule, no further runs of this recipe. Full detail:
`RESULT.md`.
