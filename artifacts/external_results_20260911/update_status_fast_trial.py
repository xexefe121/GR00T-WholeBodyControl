from pathlib import Path
from datetime import datetime,timezone
import json,shutil
base=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
doc=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911')
now=datetime.now(timezone.utc).isoformat()
note='''## Primary milestone and active run

User explicitly redirected priority: stop expanding audits/bookkeeping; decisive next milestone is a FAST controller completing an entire motion and continuous standing hold. Timing work is paused. Finish selected recovery fit then existing full walk0031569-control simulation plus250-control continuous standing hold. Preserve original source timing/native limits/hand-head/12-leg intent. No partial trajectory counts as success. Real Pico/DDS/robot remains prohibited.

ONE actual D3 fit RUNNING in direct_target_width251_student_v1, wrapper24844/child5864, root sole owner. Configuration10000 updates81000->91000/Adam16000->26000/lambda0.2; no expansion/reset/new schedule. Request7d971b56d3746e582f5d9b014c45d98e07c4e71aa7ecf229ff1a86af281ebdb5, frozen50b3958b202cc296505223ece86a1d9435915c037250bc234412565f1736e3c3, launchfa899280f0ef2baeeefbe517c8e29b49e89c11267bb8c1201b6791a8cd6125a1, clearance9d15e35b9da8a241b9eeb054c80be501f22cd74f36e792a130b49ec4e6241de6, concrete7e886c73a4ab594feac0ebc7593d9d474d68457cfd6bb4705af676b81f86d361. 439 input/39 source/480 launch/483 final process pins. Existing run_fit_durable_v1.ps1 and dispatch_fit.ps1 remain frozen. Never relaunch. Inspect fit/progress.json and fit_process_v1/exit.json. On completion run existing verify_completed.py once.

Existing fit validator adapted and ready: direct_target_width251_fit_independent_v1/source_preparation.json0c44f938053a6ccc939c72d4371926760dca4bcd45aca170e8f04dad2d38f6a2; source_root_review.jsonf2981a87830c7ee2020b0ee2d2683b4fb7041422170ed14c04147b70079b100d.21 source modules, seven original numerical helpers unchanged,86 synthetic checks. Initial test fixtures lacked new recovery roles; fixed. Root also corrected process checker to match actual owner's dispatch_attempt and four log output subjects, with12 rejection cases. No actual saved-fit audit yet. Four helpers prepare_audit_request.py/freeze_launch.py/run_audit_durable.ps1/verify_completion.py reuse previous mechanics; request waits for completed fit owner. Validator only prerequisite of existing simulator release path; no further scope expansion.

Existing full-motion simulator prepared in direct_target_width251_evaluation_v1. Native evaluator, runtime/features, witness and physics unchanged; only endpoint/recovery release metadata adapted. Source40 (34 original exact) f772f1ae3de24698e06462494d737af04a42734434be46ca60e98ec7e8b421bc; combined source/helper reviewc1bbcecc7e14b795545e288500d93d7430b79f2cd6bd81a6cc278fd9d8e19bd3,82+41 tests. Actual runtime inventory e9e8305872a60b8bbb6f957d463004c0bccc8ad50933a8e46c712b9d72106bc7. New prepare_full_trial.py binds completed fit+validator, prepare_stage.py prepares existing witness/evaluation packet and exact concrete clearance; dispatch_stage.ps1 requires Mode and literal ClearanceSha256, hidden captured handle/CreateNew/no retry. These three helpers not yet executed; no witness/binding/evaluation output exists. Run one witness then one full1569+250 canonical trial. Don't substitute another training loss analysis for that run.

All three subagents hit account usage limits around23:30 and stopped. Root continued locally after latest user resume. No active subagent work should be assumed. No new delegation currently authorized by developer mode.

## Completed instrumented timing trial — do not repeat

Clock ONE23:21:34->23:25:08UTC, wrapper12696/child24812/Linux384/461/462, all absent. Raw/diagnostic2,12109 native returned/captured,12108 verified, strict joint bound on final step. Ownerd51bc2b9aef2641dcb70edfb395c61c3345b480b387071601de665d4a9576886; report5c8bebc02cbdfbb7a83849b13aa65114aacc2a672b8382c014f65d67857aad49.138024 complete timing spans/498GC records, no probe faults; four MJB exact. First command divergence1200,11 held controls1200..1210,56 foundation misses. Original epoch unchanged.

Matching saved audit ONE01:08:22->01:11:11UTC, wrapper22896/child6688, raw/wrapper0;3871 pins exact/processes absent. Requestdd23d2ee96356e0381d384cd8839c8aaf72cada30f9ab6503ddf3b8233706a54, launch4559e431a0ca69fc714e08fc583098f030faf8097306d1b865ae1463fdf09374, rootbfb5b0a9dc1ff5129b98a9fa616ff662f8967bb2782529311ad9a35ff386baab, clearance6eda138472b2c1f7a604bcf353360caa838dfda428c85c4d6f6ff4981e26c40d. Reporta5d3b0bad24df14ca0fd893148c79446d724a6209feb916c4fb5077d3f8c54ff, ownerbf90f9597feeae6c2932233e298599635a420b3e7fd57bab2662c8f661d2b92d.21865 saved comparisons pass; physical/timing/commands false. No new native/model calls.

Saved phase evidence locates21.117ms tick11991 stall primarily inside CAPTURE19.843816ms; largest native MJ_STEP0.588133ms. CAPTURE_OWNERSHIP tick5529 also11.569433ms. CPU counters recorded but WSL clocks must not be overinterpreted. GC overlap not yet diagnosed; no GC or control change selected. Further timing diagnosis paused for user-requested fast full-motion milestone.

'''
path=doc/'CURRENT.md';text=path.read_text(encoding='utf-8')
snapshot=base/'root_status_snapshots'/('before_fast_trial_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
snapshot.mkdir(parents=True,exist_ok=False);shutil.copyfile(path,snapshot/'CURRENT.md')
start=text.index('## Active work\n');end=text.index('## Completed recovery and independent qualification',start)
text=text[:start]+note+text[end:]
lines=text.splitlines();lines[2]='Updated '+now+'. Fast full-motion plus continuous hold is the primary milestone. Simulation not yet qualified.'
path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
with (doc/'SESSION.md').open('a',encoding='utf-8') as f:f.write('\n\n'+now+'\n'+note)
print('Active fit and fast-motion priority saved.')
