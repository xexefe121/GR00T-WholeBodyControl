from datetime import datetime, timezone
from pathlib import Path
import shutil

base = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
doc = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911')
now = datetime.now(timezone.utc)
snapshot = base / 'root_status_snapshots' / ('before_recovery_qualified_' + now.strftime('%Y%m%dT%H%M%SZ'))
snapshot.mkdir(parents=True, exist_ok=False)
shutil.copyfile(doc / 'CURRENT.md', snapshot / 'CURRENT.md')
note = f'''# Current native23 simulation work

Updated {now.isoformat()}. User requests continued work. Full 23-DOF real-time teleoperation remains unqualified. Real Pico/DDS/robot waits for simulation confidence. All original SIM_ACCEPTANCE.md limits remain unchanged.

## Active work

Instrumented clock selected ONCE. Sole dispatch/owner: pico_continuation. Root has not dispatched. Check actual clock_process dispatch/start/exit evidence before any action; never repeat. All other numerical/model/native work is paused through this run and its matching saved audit. Source/JSON preparation remains allowed. Base independent_plant_timing_integration_v1: request005fcf08f9a36bc34315eddfbe506432181d6ce5c2c5548b60153d35dcfaff80; launch214221511d1a722061c4426317cb3555b3437b08377a137ca1e8b36bbad4db08; root concrete0785fedce352a9935499d97c2b851eff89e3841e49141d1cf70f9de30254bff8; actual clearance6238ac7792bcaed04b68be2bb6fe3ded7d2cd34457d016e9813735083d67ab4c. Original 1819 controls/18190 native steps/four MJB, 500 Hz plant/50 Hz commands, fixed epoch and 555+5 outer watchdog retained. Timing sidecars measure wall/thread/process CPU and GC overlap without changing control arithmetic. This is recorded-command timing evidence, zero learned-model inference.

Next D3 training configuration selected, execution NOT yet selected: direct_target_width251_student_v1/selected_protocol.json1cade1a88df2324511ca2b9599b69261e9a9b7a72ba34c540b09b8ac10896128. Keep all width512 weights/Adam/RNG from ordinary81000/Adam16000; 10000 updates to91000/26000, same prior250-step ramp then9750 cosine schedule, new separate three-phase1018-row term lambda0.2. Preserve old N15/P9/F54 data, exact original 10000x864 response schedule and old coefficient1.8188207859141674. No model widening or optimizer reset. Source39 prep658ee9edccc99f9b1e8c8124d827ccff1db1ecef53ac08d6631e2ca869d42420, root review15731f7dedf2cf3e05b1c331b519745c756d1e82a70ec727af4c75f8a024beec/36 synthetic tests; independent review10e28f3aa1924110df80ad2a3d5d0d5933c2ccae306ad82531d871eec77fe48c. Reviewer prepares exact request/freezer/launcher/owner; expert prepares saved-fit auditor. Actual fit waits for clock plus saved audit completion and concrete review.

## Completed recovery and independent qualification — never repeat

direct_target_width512_expert_recovery_v2 completed22:57:06UTC, one run started22:16:49.270UTC. Owner3160652e153c08769671a639f569449d1ed2a776f83ddffe905dd4d3f65aa7e1, raw/wrapper0, all88 pins exact, Windows/Linux processes absent. Preserved251-control prefix plus1318 new main and250 continuous hold =1569 main+250 hold. New actual native15680; full combined18190. Fresh MPC204 solves/1018 controls; terminal300 BFM plus250 BFM hold. No later student state labels claimed from this one recovery.

Root independent original main physics9904a031f8197b5414ee08e43782e816f5aeecf40b05b8a8f8451e7be664b239: all15690 native steps/seven fields exact/PASS. Main intent7d5cc86822a609d2c7d443072c4abf009bd1176caddebe93b362f05513746b3a: all819 source frames and quiet window PASS. Hold physics1d963fcf4635bc53b0c9fd1df3939a969ee0c46ea66e3608504c43f540672856:2500 steps/seven fields exact/PASS. Hold intent78eca0bddb2782755b7dbaa7b1fc26cffddf7251985957312b00a733f114e038: continuous250 quiet PASS. Reports under direct_target_width251_expert_main_physics_v1, main_intent_v1, hold_physics_v1, hold_intent_v1 respectively. Original WSL intent attempt failed before body because inherited import required Torch; preserved raw1/zero trace or native reads. Successful intent used Windows Python310 and repository PYTHONPATH. Physics/collector/consistency use WSL Python311 NumPy1.26.4. No successful stage should run again.

Recovery source root position p95.10687m, yaw9.47179deg, relative feet.10135/.08855m, leg RMSE.08707. Quiet root speed p95.001027m/s, joint speed p95.00493rad/s; maximum native speed fraction.75850, no warnings/range excess. Slow expert success does not qualify fast student or live input.

Saved first-target comparison completed once: direct_target_width251_first_target_comparison_v1/report.jsonc38c78da4288f6e9bf6c4902c0758426df68da8cffea433083b45313e3d9a94c. At exact actual pre251 full state/history/prior, fresh expert target differs from old fixed map by RMSE.12394156rad, from student.13363636rad. Student differs from old fixed map.02775804rad. This compares different teacher procedures; no isolated causal claim.

## Completed data collection and consistency — never repeat

Qualified collector actual_v1 completed0, 1018 rows251..1268:99 acquisition/819 source/100 return. Collection qualification14589ceadec8aa4eca7e43722915a5b59c4b3052d1da8ce2f9f8e3441e29e3c1; request550b3bc3eafa372fbb3041103440bc6e89376a0102cb35b80ad208bb54f2eb4c; results report7e650844a98a3e9d5462a1a8ff45e0cbdc6847323e26b82f5b2a5685c07cf316; expert_rows.npza60257b2a6f278617feb921c9c8dcdf7a4163808db0c5473acb0e39c4be8a76b. 204 actual plan gains/58-state tangent reproduced executed targets; zero feedback clips,13 native-clipped rows. Full291 state, prior23 and incoming300 history advance once. One fresh student-state query plus1017 connected expert states; not1018 independent student-state queries. Zero new model/native/optimizer calls.

Consistency actual_v1 completed0, report2ddf917a98d77268e55fe38151a94a3208428d4fe334a613820583c450a6cbaf. All13976 old nominal+physical+new rows: zero aliases or duplicate conflicts in six tested views. Fixed72 nearest-neighbor queries over12958 old candidates: new251 nearest old physical251 has normalized full RMS.02773198 but target RMSE.12305213rad; history identical, other inputs distinct. Nearby distinct inputs and teacher procedure differences remain concerns. Full354612 secant corpus not included in alias search;72 queries are not global coverage proof. Zero model/native/optimizer calls. Literal normalization path for next fit must be own width512 fit/shared/normalization.npz, even though older copy has same e914 bytes.

Saved visual rendered once, zero stepping/model calls: direct_target_width251_recovery_visual_v1/rendered_v1/contact_sheet.png. Fixed world camera, original reference timing and full main+hold coverage; no alignment edits. Root viewed all9 panels; open_in_codex queued file tab.

## Latest failed fast model and clock — never repeat

Width81000 fit completed10000 updates and numerical parity passed. Canonical trial failed control309/sub6, right hip pitch32.112498 versus32rad/s;3096 native steps,60 learned controls,zero source controls/hold. Root physics reproduced all seven fields exactly; intent failed. Better fitted loss did not establish balance. Checkpoint PT825f86468fcc83913066a9a818f4f438d080e054e17aa4aaadc1451974b8688e, ONNX8a4c394b8836dd763d6eb1c5beac73bf49921bcd59fc5f04850b377d0e843044. Original fit/eval/semantics owner reports remain immutable.

Previous pending-result clock failed12663 native/12662 verified. Tick12099 took23.076358ms; next job1211 created1.414741ms after fixed deadline;40 misses/56 held controls then joint bound. Worker BUSY retry not exercised. Matching saved audit14659b36321c3f5b9ef825a9029e95a2388ea65c45ec4f3f24802659e8e8157a passed22963 comparisons, owner66a2af16af7a3e861122b793001371b908bdf98584c937e41d341db43091a0f9; physics/timing/command qualification false. New instrumentation aims to identify stall location; overhead remains unmeasured.

## Next steps and preserved history

Complete instrumented clock owner and exactly one matching saved audit with v2 auditor prep78d7855373d409fcd6c3a44edef433d6be520faaaf956a739483dad2b8dd6ae8/root3615d4c6df567cb917bfd7d2835d1e5d4c4db76582bbac66f2497536c0399cf3. Helpers root e1fb37c1d1833ae263b9c9c37f56a166863747440792eeb8e5a3922525c8140d. Then concrete D3 fit, saved fit audit, fresh activation witness/canonical full1569+250 and original physics/intent. Only after successful walk003 expand same fast controller to full PICO/walk002/heldoutwalk008 and real-time/fault/perturbation checks. No fast controller has qualified complete suite.

Earlier detailed status preserved in [{snapshot.name}/CURRENT.md]({(snapshot / 'CURRENT.md').as_posix()}); SESSION.md is append-only. Earlier complete ledger also remains under root_status_snapshots/before_recovery_path_fix_20260911T221115Z. Large artifacts stay on E:; dirty repository contains previous work; no reset/cleanup.
'''
(doc / 'CURRENT.md').write_text(note, encoding='utf-8')
with (doc / 'SESSION.md').open('a', encoding='utf-8') as stream:
    stream.write('\n\n' + note)
print('Current status and append-only session updated; no numerical experiment.')
