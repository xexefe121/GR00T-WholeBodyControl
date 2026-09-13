# Native23 work snapshot — 2026-09-13

## What is published

The complete upstream-derived source tree and its existing development history, all current source changes and new native23 scripts/configuration/tests, readable experiment reports, and archived large experiment outputs. External results formerly on the artifact drive are indexed under `artifacts/external_results_20260911/`.

This is a reproducible research archive, not a qualified teleoperation release. Historical reports retain their dates and failures. The latest corrected A/B run rejected continuation; the longer independent-clock run lost input and is not successful motion completion.

## Start here

- `gear_sonic/`: native23 controllers, task/reference adapters, training/evaluation, tests and model contracts.
- `gear_sonic_deploy/`: deployment and native controller code.
- `install_scripts/`: existing environment/setup scripts.
- `artifacts/onboard_inspection_20260912/RUN_CURRENT_NATIVE23_TELEOP_SIM.ps1`: current recorded-simulation launcher.
- `artifacts/onboard_inspection_20260912/CONTROLLER_MEMORY_EXPERIMENT.md`: latest implementation and measured results.
- `artifacts/teleop_six_hour_20260910/SIM_ACCEPTANCE.md`: original acceptance thresholds.
- `artifacts/external_results_20260911/onboard_factory_firmware_v1/controller_state_ab_v1/RESULT.json`: detailed latest result.
- `PROGRESS.md` and root handoff documents: earlier work and decisions; later dated results supersede earlier readiness statements.

## Download the larger results

[GitHub release](https://github.com/xexefe121/GR00T-WholeBodyControl/releases/tag/native23-work-2026-09-13) contains independent `.7z` archives and an archive manifest. Archives are grouped by original storage root and split below GitHub's per-asset size limit. `repo-artifacts-*` extract relative to the repository root; `external-results-*` extract relative to the external artifact root. Use 7-Zip to extract each independent archive. Text reports already published in Git do not need an archive download.

`publication/manifest.jsonl` lists included file paths, sizes and delivery location. `publication/exclusions.json` names excluded machine inventories, raw extracted manufacturer firmware, external links and generated caches. `publication/sanitizations.json` lists credential-related edits applied only to the public copy. Source and research originals remain unchanged locally.

Original downloaded third-party policies, runtime dependencies and datasets remain governed by their original licenses. Use the repository's existing download/setup tools and preserve upstream license notices. Extracted manufacturer firmware is not distributed here; firmware-dependent experiments require the owner's existing local installation/files. This archive does not claim those experiments run from a bare clone alone.

## Environment and execution

Existing launchers target Windows plus WSL and record the original `Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof` and `E:/codex-artifacts/sonic23_teleop_resume_20260911` roots. Restore those directory layouts or adjust the documented launcher paths before running. Native simulation uses MuJoCo 3.2.3, 500 Hz physics and 50 Hz control; training/runtime setup and manifests identify the specific environments. Estimated-feedback and live-Pico modes are implemented but not qualified.

Read-only SSH inspection helpers now require `UNITREE_SSH_PASSWORD` in the environment. No robot connection or command is made by this publication. Motor tests require a separate supervised test window.

## Status of archive upload

The public release and `publication/archive_manifest.json` record completed assets. Until that manifest says `complete`, some large downloads may still be uploading. The code and reports can be reviewed independently.
