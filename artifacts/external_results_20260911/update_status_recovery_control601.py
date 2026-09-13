import json
from datetime import datetime,timezone
from pathlib import Path
NEW=Path(__file__).resolve().parent
DOC=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911')
note='''## Active checkpoint and completed preparations

Recovery_v2 remains ACTIVE; latest reported control601, no failure, solver7.40s per five-control window. Wrapper13580/WSL28320/Linux371 are the selected run. Do not launch a duplicate. Full source/return/hold and owner completion remain pending. No first-target comparison or independent replay has run for this recovery.

Conditional collector source9 files/4 unchanged,28 producer+28 root fake tests PASS. Source preparation6704baa14dd451db12269ab58449e1575ee78c31bc31f4b1ed3da416b1d4cd7e; root direct_target_width251_collection_root_review_v1/review.json8b1aeea19361feb5bde70625b003b0f7050b1b858655786b7b5c776a13c9b285. Actual1018-row collection remains gated on completed owner, four independent full/hold physics/intent reports and explicit root collection selection.

Root helpers in direct_target_width251_followup_design_v1: run_saved_comparison.py binds completed owner and old semantics outputs before comparing actual first target; prepare_audit_commands.py binds owner/actual traces and prepares existing original main/hold physics+intent commands, including existing zero-step hold-fixture converter. Both source-only; no actual invocation yet. Use fresh recovery_v2 canonical owner_completion.json, never v1 failed owner. Intent needs PYTHONPATH=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof for original hand-frame module; physics inserts frozen recovery source itself.

Conditional DAgger source/text design direct_target_width251_dagger_design_v1/report_v2.json868f6420fdf32e8da350c271288195485c996ffc2dcbff16e86fc002e766f7b5. Keep512 architecture, frozen1323 normalization, all81000 weights/moments(step16000)/RNG; do not run old256-expansion initializer again. Add a separate1018-row loss and diagnostics without changing old N15/P9/F54 arrays/index maps/coefficient. Update count/LR/new coefficient remain unselected. Reviewer preparing pure saved consistency checker:13976 new+old nominal/physical raw1323 rows, fixed72 proximity queries; no actual arrays/model work now.

Timing saved auditor v1 source preparation77710e582160f9ab345268792fd3d1a776a8afc48045d410a451b4cbaa379fc9 is NOT root-cleared. Root independent synthetic proof direct_timing gap report under independent_timing_saved_audit_root_review_v1/v1_gap_proof.json demonstrated4 accepted errors: verifier-attempt reported as return, sequential sibling wall overlap, child thread CPU beyond parent end, child process CPU beyond parent end. Pico preserves v1 and prepares v2 with regressions. Integrated producer32 remains root-clearedfdccbf0e; no actual instrumented clock selected. Pico also prepares durable helpers source-only. Actual clock waits for active recovery and independent native audits to finish.

'''
path=DOC/'CURRENT.md';body=path.read_text(encoding='utf-8');anchor='## Active work\n'
assert anchor in body
body=body.replace(anchor,note+anchor,1);path.write_text(body,encoding='utf-8')
with (DOC/'SESSION.md').open('a',encoding='utf-8') as f:f.write('\n\n'+datetime.now(timezone.utc).isoformat()+'\n'+note)
print(json.dumps(dict(updated=str(path),recovery_active=True)))
