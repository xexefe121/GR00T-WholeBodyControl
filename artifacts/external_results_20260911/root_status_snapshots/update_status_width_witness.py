from pathlib import Path
from datetime import datetime,timezone
import json
NEW=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
ART=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911')
now=datetime.now(timezone.utc).isoformat()
for name in ('CURRENT.md','SIM_RESULT.md'):
    with (NEW/'root_status_snapshots'/name.replace('.md','_before_width_witness.md')).open('x',encoding='utf-8') as f:f.write((ART/name).read_text(encoding='utf-8-sig'))
current=f'''# Current native23 simulation work

Updated {now}. User said continue. Work remains active; fast full-body simulation remains unqualified. Real Pico/DDS/robot waits. NEW=E:/codex-artifacts/sonic23_teleop_resume_20260911. Prior full lineage preserved in root_status_snapshots/CURRENT_before_width_witness.md.

## Active ONE width81000 runtime witness; root sole dispatch owner

NEW/direct_target_causal_width512_evaluation_v1/witness_process. Started20:46:54UTC hidden wrapper26084, child27516, captured handles. Exactly one WSL head call authorized, zero physics/BFM. Do not duplicate. Root dispatch_witness.ps1 and dispatch.json preserve original attempt. Binding0d3b89d4917c62a5dc889753f4c111371d7197233cd4425cfbc6c5a2b54ded04/5227pins; launch8c84b96b2d3ba41acd83074f1d89f96ebbfbac194881bd76ce15af5ab7b2c526/5232pins; root concrete dc82efa6cc25c4aecdcd12de9e6638eae0ee34b7f2dc665ab2cbb9ba2a82d828; clearance47ee94cb1215f9f523bff8f2ec7fd1364a12c8cc33279c7f2fe39c3598a6d828. Actual PS5.1 launchers parsed, all pins checked. Await result/raw exit/posthashes/PID absence/owner before canonical binding.

Root final release2dd758b45e2e64294c275a969d05acc6cbeae74ddae509c63c2fa8e40f9e72ee selects fixed endpoint and original1569 main controls plus conditional250 hold. Configurationc1d31a3b666585c326dbe2c691acd21cdbbb488a980bee697284bbd269b6792e. Source38modules/34unchanged, prep5703e68e439963c271dd79d5d0f30aaa3be9683f2d28096502aedb3747b96962, root121tests/independent121tests db2a997d44634cb59d7d4bd92408fcf361323d143c6109024b457e5bd4a458de. Eight helpers root41tests/independent41tests647de5513de061fbf231aad4d1d6e728d1b606e38252f946f11f39cfacc7fe36, four old files byte-identical. Inventory5194files6a6f771154e3cbbc7660713f77a12df6592fcf39f0962c34739c58101cfc7f1e, same full291fixture. Physics evaluator061e2a3145c6ed273bf7ac915efec0f1866ba712ccb8ba571e8c08417243d970 unchanged. No canonical execution yet. Root functions width81000PhysicsAuditStarted/IntentAuditStarted=false, prepared commands only.

## Width512 training COMPLETE PASS; never repeat

NEW/direct_target_causal_width512_student_v1. ONE20:19:37→20:32:20UTC wrapper23928/child25188, both absent/raw0/wrapper0. Reportb5979023fe9bceb14e0d54cf04bf727fbc413d5a1443b39018e149443b418de6; owner18c66ebdb2af7138858c08d012c5d3cc7819c1bb2bb5a5b523e488c35e4ed6a2. All10000updates, ordinary81000/optimizer16000,30000 training forwards/146860000rows. All five diagnostic backends each1437calls/367570rows. Initial outputs byte-exact71000; final same-weight FP64 worst5.2015483475997826e-8rad against1e-5 gate. FinalGPU32 drift diagnostic only. PT825f86468fcc83913066a9a818f4f438d080e054e17aa4aaadc1451974b8688e; ONNX8a4c394b8836dd763d6eb1c5beac73bf49921bcd59fc5f04850b377d0e843044. Training requestf78bc57dc24af30fca2f7d20a05bd82ddf78e51158ebe1f41f22b85784481248; frozen4cb56ebc741ac40f6de52e83cba8f718a7db78f07e1d02a1c25c02902d78e939. Exact old256 blocks/moments/context/normalization, added512 width and250+9750 LR ramp; cannot uniquely attribute improvement to capacity.

GPU32 nominal objective.000313900699742→.0000952567682058; balanced response.000256651839660→.000207212140057; physical.0000802511313294→.0000451063917191; balanced total.000860955531787→.000517244907355. Original response.000202364100327→.000178764411567. No closed-loop claim from these metrics.

## Independent saved fit audit COMPLETE PASS; never repeat

NEW/direct_target_width512_fit_independent_v1. Sourceprepd252b0a8875636601d90bf538f342097043bce86666d6ddb23c4d708e00f0d2e,16sources/9oldbyte-identical+4helpers. Root77synthetic tests/source review7de83d50734c8cce58ac78d1b9c1352422f910e63b8d400a5fe4a1c6a6997419. Request7e0d079af78b680331ec83cb294073fe8ae2f3382b4fa2b3e6a2b89d10ca397a; launch e53a061fb672a1bb6e1c518cf284115001abac187d4e17226dccdb37e71bc864/41pins; rootconcrete40a56398b26d3447a20016f6ef60be3ad5570b1a37da999ff0f6eb4274b404a0. ONE20:42:44→20:43:57UTC wrapper27952/child12196 absent/raw0/wrapper0. Report3656f44031d9362fdbabeab69d2423266839b88662cbe46b79953a977dc56cd7, owner24b9ba22d572874a5e4bd4c97a282e9444126231807ff70d526312303eb12122.66,929 checks pass; all16release subjects/schedules/context/expansion/moments/RNG/numerical algebra exact. No model/ORT/gradient/optimizer/native calls. Saved audit cannot establish model-output authenticity beyond saved algebra and provenance. Reviewer now preparing width81000 saved-semantics source only; no actual trace assumed.

## Clock result retry source ready; no new clock run

NEW/independent_plant_pending_result_v1/source_preparation.json88080d6c84025b7589a381a1d5d3dc5f170acdb69b20c95f31123f3c6278b4ad.24sources/16prior byte-identical; explicit original Job.deadline_ns, immutable pending result,20 total attempts/once existing1ms worker loop/BUSY only; no recomputation/rebase, retained two-slot owned batch, uncertain result fail closed.67 producer and67 independent synthetic tests passed. Independent review NEW/independent_pending_result_source_review_v1/review.json b017dbb753fe2c8a0370288de01eb9e52275e2e5a365b54582da20cb15700640. Pico preparing retry-aware saved auditor source. No request, actual worker, clock or native run selected yet. Original15plant timing misses remain separate from channel BUSY loss.

Prior pending-publication clock COMPLETE FAILED:6824attempted/returned/captured,6823verified; command656missed,27controls held655, strict joint bound. Four job BUSY retries recovered. Workerresult656 BUSY had17.413713ms remaining and was dropped. Reportdf8d562e18508b24372b81a7ecab6e8e0d53e195562078dc23ff6a45d54a89f6, ownera51d7f88bff439a23c696ad48fa6cff5c8ba035cae58820cd1ac2c7f4ccf8b62.15captured deadline misses/16outer, extraouter-only5314; maxruntime debt5,11366unexecuted. Saved audit6d53378fdb7b439487728d5b8a56796ee6343f068dc42a0f8558b3613e3e0d24 and ONLY correct owner_completion_dispatch_v3.jsonbaef944c8c056494c46f8eed67e58ec36ae092f379589e7ce5493d0a053cc775 complete. Failed pre-body dispatch and owner-parser versions preserved; no actual audit repeat.

## Prior simulation failure and acceptance

71000canonical failedc287/sub9,t5.758s,left_hip_yaw_joint-32.225616684rad/s vs32. Exactly2879steps/288issued/38learned/source0/hold0. Root independent replay all2879steps/sevenfields bit-exact. Saved semantics6658checks passed; first feature departure251/first clipping253,23of38 clipped; query250RMSE.047995195rad. All prior failed models, traces, audits and process records preserved; no reruns.

Original acceptance remains: same fast controller on fullPICO,walk002,walk003,held-outwalk008; source root/yaw/feet/legs/hands/head intent, every2msnative position/speed/effort/body/clock limits, final3squiet andcontinuous5shold. Then independent500Hzplant/50Hzpolicy, received-only inputs, faults/stops/rearm/perturbations. No crop/retime/referenceedit/rootforces/limitrelaxation. Prepared preview and ground-truth root still disclosed. Existing slow offline expert fullPICO6530+250,walk0021417+250,walk0031569+250,walk0081114 passed; video NEW/pico_qualified_full_video_v1/full_pico_and_continuous_hold.fixed_world.mp4. This does not qualify fast live teleoperation.
'''
(ART/'CURRENT.md').write_text(current,encoding='utf-8')
(ART/'SIM_RESULT.md').write_text(f'''# Native23 simulation result

Updated {now}. Fast full-body teleoperation remains unqualified.

Width512 model completed10,000updates. Independent saved-data audit passed66,929checks; all367570initial predictions matched71000 exactly. Export worst numerical difference5.20e-8rad passed1e-5gate. Saved nominal error fell70%, balanced response19%, physical examples44%. One runtime output witness is active, then original fullwalk003 physical trial is next. No canonical result yet.

Previous71000model failed5.758s at hip-yaw speed32.226rad/s against32limit. Independent physical replay and saved input/history audit completed. Stability cannot be inferred from fitting loss.

Clock BUSY result-loss fix passed67 independent synthetic tests. Original command deadline retained, bounded immutable result retries. Matching saved auditor source is being prepared; no new clock run. Separate plant deadline misses remain unresolved.

Slow offline experts already passed full recordedPICO,walk002,walk003,walk008. Full recorded PICO video: E:/codex-artifacts/sonic23_teleop_resume_20260911/pico_qualified_full_video_v1/full_pico_and_continuous_hold.fixed_world.mp4. Fast-controller full-suite, independent500Hz/50Hz timing, received-only input, fault/rearm tests remain required before real Pico/robot work. Prepared preview and ground-truth root assumptions remain disclosed.
''',encoding='utf-8')
with (ART/'SESSION.md').open('a',encoding='utf-8') as f:f.write(f'\n\n{now}: Width81000 fit complete b5979023/owner18c66ebd; independent saved audit66929checks3656f440/owner24b9ba22. Root selected one runtime witness20:46:54wrapper26084/child27516; binding0d3b89d4/launch8c84b96b/clearance47ee94cb. No canonical yet. Clock immutable result retry source67independent tests b017dbb7; saved auditor source underway/no clock run.\n')
print(json.dumps(dict(updated_utc=now,witness_active=True,training_complete=True,fit_audit_complete=True,qualification=False)))
