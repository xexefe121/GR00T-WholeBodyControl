from pathlib import Path
from datetime import datetime,timezone
NEW=Path(__file__).resolve().parent.parent;ART=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911')
p=ART/'CURRENT.md';s=p.read_text(encoding='utf-8-sig')
with (Path(__file__).parent/'CURRENT_before_audit_v3.md').open('x',encoding='utf-8') as f:f.write(s)
start=s.index('First saved-fit audit COMPLETE FAILED');end=s.index('Evaluator prepared under',start)
s=s[:start]+'''Saved-fit audit v3 ACTIVE, one selected dispatch19:31:48UTC wrapper23048, reviewer soleowner. NEW/direct_target_response_balanced_fit_independent_v3 request7a09feb170fc2f0423d55ce471a82bc07d00f0e5fcce6853a1d428462879ffc4; launch0f5b558c6e08bf50923fe8ee03892108d9914ff79ddf6e9be99ed82de146e81e/38pins; rootconcrete9b8545417fa6263f9fd7fcb0df82ec7bb7f5a1316f3ff8bab0b204cccb3b8ce6; rootsource532a93fc2e8e44aec75a0792802537877f84643243d2f3abf8ec0e271fff0c8b/50tests. Do not duplicate. Two previous failed saved audits are preserved: v1 wrong rule string at1948checks (reporte6013ad7/owner83a5e0bb), v2 redundant CPU-f32 mean ordering check (report48c3bc8e/owner6395cc6c). Rule diagnosisd912a020; reduction diagnosiscbbef016. Three of3000 nominal rows differed by up to4ULP vs NumPy-f32, but original CUDA-vs-f64 mean3e-7 passes allrows and exact positive-sum gamma16bound holds. V3 removes only the redundant CPU-f32 check, retaining original nominal/model gates. No training/checkpoint/prediction changes or new model/native calls.

'''+s[end:]
s=s.replace('Expert adapts eight helpers to fitv2 with old helper version preserved.','Eight-helper review complete b95fa1160498af88cf3fe38c80aaf12b8707ac3d18f25ed69fdfe54c9920d16f/41tests. Launcher differs only139CRLF-to-LF bytes; otherthree unchanged helpers byteexact. Original mistaken byteexact checker preserved, no evaluator source mutation. Root final release writer prepared, not executed until audit and owner pass.')
s+='''
## Prepared next candidate and clock review

Width512 source-only preparation NEW/direct_target_causal_width512_preparation_v1/source_preparation.json33c12592a71a709b78e0843219541e6cfc9124ba25027ff103820c77c7617ac2, proposalc4882558c35c6281f19e945277490e1e416770b2f21e997729d20b6a54062aab.27syntheticCPUtests pass; no actual checkpoint loads/forwards/native. Preserves old256 contractions, adds random incoming/zero outgoing blocks, six expanded warm moments step6000/newzero slots, preserved RNG. Proposed10000updates→ordinary81000/optimizer16000,250LRramp1e-6→1e-5 then9750cosine→1e-6; unselected. Width and LR change cannot isolate capacity causation. ActualCUDA/FP64/batch1/fullphysics needed. Saved convergence report0f7d8480441c022efb1fe9095d495e27ab689e9259d56f510c13188b55fd7bb5 shows entryRMSE worsened .04737766→.04799527rad, firstjoint error .153246rad, despite aggregate gains;98.1%balancedresponse errorodd.

Retry-aware saved auditor v2 rootPASS81c4090dd6035fc4b39064fd98e7412af87e2d0355a4a3abd260d19ffb4faae0,106tests,source17files. V1acceptedfourforgedterminal histories, allfixed andpreserved; handles explicitboundouterinterruption withoutfalse nativecredit. Pico prepares one futureclock request/helper packet; NOclockdispatch until rootreview and idle afterfit/physics. Saved timing diagnosis NEW/independent_clock_timing_diagnosis_v1/report.json71d6ea21df432918d0c2706583dd7322e19bc6ae007456d0186491cfa7887dca: firstmiss610wake26us/body2.268ms, not asleepovershoot; bodymedian.490ms/p95.974/p991.355/max2.650. Body includesnative/checks/Python/preemption, cannotascribeOScause. PendingBUSYfix alonecannotqualifytiming.
'''
p.write_text(s,encoding='utf-8')
with (ART/'SESSION.md').open('a',encoding='utf-8') as f:f.write('\n\n'+datetime.now(timezone.utc).isoformat()+': savedauditv3active wrapper23048; previous2savedcheckerfailurespreserved, numericalmodelunchanged. Width512sourceonly27tests prepared; retryauditroot106testsPASS; no canonical or newclock selected.\n')
print('CURRENT and session updated.')
