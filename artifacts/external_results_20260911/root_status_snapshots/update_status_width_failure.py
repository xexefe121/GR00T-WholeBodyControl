from pathlib import Path
from datetime import datetime,timezone
import json
NEW=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911');ART=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911')
now=datetime.now(timezone.utc).isoformat();path=ART/'CURRENT.md';s=path.read_text(encoding='utf-8-sig')
with (NEW/'root_status_snapshots/CURRENT_before_width_failure.md').open('x',encoding='utf-8') as f:f.write(s)
start=s.index('## Active ONE');end=s.index('Completed ONE runtime witness',start)
section='''## Width81000 canonical COMPLETE FAILED; root audits complete, do not repeat

NEW/direct_target_causal_width512_evaluation_v1. ONE20:54:59→20:57:49UTC wrapper21884/child27236 both absent. Raw Python0, diagnostic/wrapper2, errornull, all5243pre/post/current pins exact. Owner evaluation_completion_verification.json0caad5d8a7574bfd0a082be64b1f17c8cfd0be38360ef28bad767adfe6b3552a. Main reportd8b81c75ccf16b0d9180dcd7f24b468c66a63ca6401ed644a023f616a627c5f1; traceb9061a1c9dc6aee65d713f16909a3135f072d94a5bc20af750b5958cc493629d. Strict failurecontrol309/sub6,t6.192s: right_hip_pitch_joint(index6)dq32.112498019403844rad/s vs32, ratio1.0035155631063701; q.5642359729194842. Exactly3096native/309full+6partial/310issued/60learned; startup250BFMbackward+250actor, learnedBFM0/forbidden0/source0/hold0. Full291/prefix250/query250input/outputparity passed. No hold trace. Allphase policy p50/p95/max6.65255/8.67719/13.8289ms,0misses20ms; this is not independent plant timing qualification.

Root ONE independent physics NEW/direct_target_width512_independent_physics_v1/report.jsond20ba43f42227258a26b293edef6134cc7772391cbeac4cdfa87f5127d31da20 reproduced all3096steps/sevenfields bit-exact, zero steps beyond recorded failure. Exit1 is expected preserved strictfailure. Root saved intent NEW/direct_target_width512_independent_intent_v1/report.json9d89c2456658a1df603a5b413786846a34a5f92fa140b82ebeda1d93db1e83af complete, sourcefalse/quietfalse,0dynamics. Root functions width81000PhysicsAuditStarted/IntentAuditStarted=true and both completed; never repeat. No model/native work active now. Agents may resume synthetic tests; no new clock or fit selected.

Canonical bindinge5958d057927523ebb4d1e5870f7dca5edc03f2723a72f52a53766f13a68b42d/5238; launch9a70376c12c620c52aba46885280411e03ed0ab525a6a71f6b54656b2914a6a9/5243; independentreview84e6c263d8cfe537083d9460d1872c7e6886e7756d30593654a82858af04507d; clearanceb83640bc15fc8c3cf67e779a07eb9a926eb53c04a29504267c5e1f376be10f38. Root dispatch_evaluation.ps1/dispatch.json and captured handles preserve one attempt.

Next required: actual width saved-semantics request/concrete/audit/owner. Reviewer has exact owner, rootphysics and intent subjects plus source clearance0d92c331; preparing request after bounded read of prior fixed-map/on-policy relabeling experiments. Actual failure-state expert replans not yet selected. Width improved supervised fit and entry error but failed stability; use actual feedback/history/state departures to choose next change. All prior failures preserved.

'''
s=s[:start]+section+s[end:]
lines=s.splitlines();lines[2]='Updated '+now+'. User said continue. Fast full-body simulation remains unqualified; real Pico/DDS/robot waits. NEW=E:/codex-artifacts/sonic23_teleop_resume_20260911. Previous complete status preserved in root_status_snapshots.'
path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
sim=ART/'SIM_RESULT.md';old=sim.read_text(encoding='utf-8-sig')
with (NEW/'root_status_snapshots/SIM_RESULT_before_width_failure.md').open('x',encoding='utf-8') as f:f.write(old)
sim.write_text(f'''# Native23 simulation result

Updated {now}. Fast full-body teleoperation remains unqualified.

Width512 model completed10,000updates; saved-fit audit passed66,929checks. All367570initial predictions matched71000 exactly; export worst difference5.20e-8rad passed1e-5gate. Nominal error fell70%, balanced response19%, physical examples44%; first takeover targetRMSE improved.048→.018rad.

Fullwalk003 trial still failed6.192s: right hip pitch32.1125rad/s against32limit. Exactly3096physicssteps/310commands/60learned actions; source motion and quiet hold not reached. Independent replay reproduced every saved step exactly. Full control-cycle timing p95=8.68ms,max13.83ms,zero20msmisses. This is a controller-stability failure; timing alone does not qualify independent500Hzplant operation. Saved history/feedback audit next, then evidence-based assessment of actual failed-state expert relabeling.

Clock result retry source passed67independenttests. Matching saved auditor is being strengthened to enforce iteration order across successive results; earlier source preserved. No new clock run. Original plant deadline misses remain unresolved.

Slow offline experts already passed full recordedPICO,walk002,walk003,walk008. Video E:/codex-artifacts/sonic23_teleop_resume_20260911/pico_qualified_full_video_v1/full_pico_and_continuous_hold.fixed_world.mp4. Same fast-controller fullsuite, independent500Hz/50Hz timing, received-only input, fault/rearm tests remain required before real Pico/robot. Prepared preview and ground-truth root assumptions remain disclosed.
''',encoding='utf-8')
with (ART/'SESSION.md').open('a',encoding='utf-8') as f:f.write(f'\n\n{now}: Width81000canonical COMPLETE FAILEDc309/sub6/right_hip_pitch32.112498>32,3096steps/60learned/source0hold0. Owner0caad5d8 all5243pins/raw0/diag2/PIDsabsent; rootphysicsd20ba43f exact3096steps, intent9d89c245 source/quietfalse. No repetitions. Savedsemantics request and onpolicy-coverage diagnosis next. Pendingresult auditorv1 preserved, v2 global cross-job iteration order fix under review.\n')
print(json.dumps(dict(updated_utc=now,canonical_complete=True,physics_audit_complete=True,intent_complete=True,qualified=False)))
