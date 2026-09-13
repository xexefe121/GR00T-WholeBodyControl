# Native23 Pico preview repair

Simulator-only implementation, source, reports, traces and videos. This update builds on the full-work branch.

- Strict rejection stops the simulator before an unacceptable preview target is published.
- Fresh older baseline completed 160.600 seconds, including 30 seconds standing, with timing passing in that run. Full-body tracking still failed.
- Fresh newer run stopped at 63.392 seconds with no joint-limit violation. Rejected command was never applied. One physics step finished late; terminal standing was not reached.
- 21 regression tests, two native rejection cases and 12 native/Python comparison cases passed. All 31,830 archived physics steps were reproduced exactly from recorded applied commands.
- Live teleop and robot execution remain unqualified. No robot actions were performed.

[Download reports, full traces and videos](https://github.com/xexefe121/GR00T-WholeBodyControl/releases/tag/native23-preview-repair-2026-09-13). The evidence ZIP includes captured diagnostics and implementation sources; videos are separate release assets. Existing checkpoints remain in the earlier full-work release.

[Summary](report.json) | [Detailed outcome](OUTCOME.md) | [Older result](older_v1/report.json) | [Newer result](strict_newer_v1/report.json) | [Matched-prefix comparison](matched_prefix.json)

Run the pinned baseline using `artifacts/onboard_inspection_20260912/RUN_PICO_PREVIEW_REPAIR.ps1`; pass `-Case strict-newer` for the guarded newer candidate. Existing local artifacts and pinned WSL MuJoCo 3.2.3 runtime are required.
