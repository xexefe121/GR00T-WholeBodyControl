"""Preserve preceding handoff and record the selected fresh split study."""
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent
DOC=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911/CURRENT.md')
old=DOC.read_text(encoding='utf-8-sig')
snapshot=BASE/'CURRENT_before_split_active.md'
with snapshot.open('x',encoding='utf-8') as f:f.write(old)
start=old.index('## Active work');end=old.index('## Latest controller result:')
section='''## Active work

ONE fresh split-first-layer matched pair selected and dispatched17:29:43UTC, hidden wrapper28060. NEW/direct_target_causal_context_study_v2, process fit_process_v2. Expert owns dispatch/monitor/owner3; DO NOT DUPLICATE. At this update child not yet reported. Request5a09d441f47e58f7cde8301aee72c23d6d0432c09cbdd37f49e4e6c3ac079cfe, frozen de1f083af0bbcda9dead1e3714e3c586a9e564bee26879072006c27c7ad2c050, launcher6f114a44e4958598271379ce57be5f553e28ff2c078bfef3da8d3c7759627c5d. Root review NEW/direct_target_context_split_launch_root_review_v1/review.json56b5ebed04b740e2a911812d2ac4ccabdeaa670ecff6616114d9a6c1fdfca87f; actual clearanceabd16ab09dace711300f645654485c95218711611c30e801b5a02d7037e9b4d0. All285input+20source pins exact. GPUPY E:/codex_sonic_runtime/direct_target_gpu_20260911/venv/Scripts/python.exe.

Source preparationdfb862c426b1ab7e32aae39a42a63bc9c72e5a8744c476e1d9dc5a2d1de27468, actual independent source reviewf08211f05632207c27d43fda8d485b0c295f0d3349b156ddf5b82317e2f940be, independent split math17tests7819d0d5c19f6466bd4a2328d9390fea83b15c9ee474e3bb33ca75873c013087, owner25syntheticPASS.20files=17unchanged+context_model split+driver disclosure changes+newtest. Same1323->256ELU->256ELU->23 six parameter tensors. Torch32 first layer computes contiguous old1000 linear+bias plus contiguous323 linear without bias, then sameELU. Monolithic FP64 promoted model/export remains unchanged and fully parity-gated. All actual initial and final1e-5rad gates unchanged; synthetic identity is not a substitute.

Unchanged fixed pair: blinded then causal, each3000updates from sameordinary65000 PT8a1b67e09285a77910d62dd1b2214c8d3684004e55a82041544e750006b29a0a/RNG, original1000norm andweights,323newcolumns0. Append incoming prior23+pre-update H300; blinded rawcontext=meanf32→normalized0. Samefirst3000savedfull58schedules, freshAdamW, inclusivecosine1e-5->1e-6, coefficient1.8188207859141674, no calibration.44.058Mtrainingrows/condition;88.116Mpair. Fixedordinary68000 endpoints andfivecompletebackendpasses/condition. Completed context data proof f5add6c31a3c3aec9b215d837d86e62a8a9bffcac4575d096d24012296c04583 reused without rerun:9899nominaltransitions/3054physicalprior+H/alloriginal1000prefixesexact. No controller endpoint selected.

Previous context study_v1 stopped beforeupdates and MUST NOT REPEAT. Firstwrapper12564 failedbeforePython duePS5.1 Decimal-vs-Double; zero task calls, preserved fit_process/failed_launch_attempt_v1. Correctedwrapper12832/Python3004 started17:16:48 andexited1at17:18:04, allpins exact/PIDsabsent. Wider1323monolithicGPU32 initial drift exceeded1e-5: nominal1.0819143341223025e-5,full_state1.0403096695199565e-5,physical7.739848697951857e-6.1437initialmodelcalls/367570rows,0trainingforwards/updates/native;causalneverstarted. Owner3accounting5cdbd6c76fdf24db13e029739b238be02a617a84ace16437d3bedbbc6f7b93a3; root saved failureaudit NEW/direct_target_context_initial_failure_root_v1/report.jsondf108b650256888562d55c141feda95b5c37fb63063f00d03cffdb31d790cc3e independently reproduced all drift/ledger and initial=failed actor/optimizer/RNG. Auditorcalls0 distinct from producer1437. No completepairedreport orrelease.

Reviewer owns eventual saved-pair audit after BOTH newfitreports andowner3 complete, plusrootv3sourceclearance. NEW/direct_target_context_pair_fit_independent_v1/source_draft_v3, preparation396d0123aae97b85084e4653a46fe5f36397951cb1eb4c340b655d3732b47d95, delta bfecd1e295bc6a9d1bc195124d51e898d91bec9afcc4f2e1535146d577d64c0e. Onlystudy_v2 path/filenames/newsplit metadata;6math/graph/test sources unchanged,16syntheticPASS/PSASTPASS. Root priorv2sourceclear ee211499592f0e26c4189aee44cb14821d7b9075eacf69b1f51de5b09bcc2fc8 androot16tests; finalv3delta review pending. No actual saved-audit request/launch yet. Authoritative study_v2 verify_completed_v3.py writes owner_completion_verification_v3.json andboth18-role condition views aftercomplete; redundant old prepare_owner_views.py excluded.

Fresh evaluation namespace NEW/direct_target_causal_context_evaluation_v2 prepared: all36runtimefiles exact oldreviewedcontextv1;7of8helpers exact, onlyfreeze_final_package.FIT pointsstudy_v2. Source/helper combinedreview NEW/direct_target_context_namespace_review_v2/review.json949b5124b0f452a917c87660a79f294e5276a8b20bc3da4c2a516d6d88dd3a29 passedtrue/exact36+8maps, newhelperprepf1dd5ab512a28f8180df3b57969b69b80a2693d2590b90ac9b0d981211713aff. Runtimeinventory4296f2429a3483797c90589213f466e6912801bb1ee05f4f2c158caa69c7b07f all5192pins rehashed. Reuse prior42helper/76runtime tests withunchangedsource evidence. No endpoint/binding/witness/native selected. OriginalincomingH/prior,query250/full291,strict2ms,1569+conditional250hold andterminalBFM unchanged. Rootrelease requires18subjects, completedpairedfit/export/audit andactualconditionowner.

'''
new=old[:start]+section+old[end:]
new=new.replace('Updated 2026-09-11 17:19 UTC.', 'Updated '+datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')+'.',1)
DOC.write_text(new,encoding='utf-8')
session=DOC.with_name('SESSION.md')
with session.open('a',encoding='utf-8') as f:f.write('\n\n'+datetime.now(timezone.utc).isoformat()+' — Split context continuation\n\n'+section)
print(str(DOC))
