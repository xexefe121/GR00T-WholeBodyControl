from pathlib import Path
from datetime import datetime, timezone

repo = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
base = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
current = repo / 'artifacts/teleop_resume_20260911/CURRENT.md'
old = current.read_text(encoding='utf-8')
snapshot = base / 'root_status_snapshots/CURRENT_before_context_ready.md'
with snapshot.open('x', encoding='utf-8') as f:
    f.write(old)
now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
prefix = f'''# Current native23 simulation work

Updated {now}. User says continue; continue autonomously. No fast full-body controller qualified. Real Pico/DDS/robot remains outside simulation stage. All paths prefixed NEW mean E:/codex-artifacts/sonic23_teleop_resume_20260911. Previous current file preserved in NEW/root_status_snapshots/CURRENT_before_context_ready.md.

## Current selection and active work

Matched context study_v2 COMPLETE. Both blinded and causal fixed ordinary68000 endpoints completed 3000 updates each: 6000 updates, 88,116,000 training rows and 18,000 training forwards total. Initial outputs are byte exact to ordinary65000 across all 367570 saved examples; all five complete diagnostic backends and both export gates pass. Wrapper28060/Python5052 exited0 and are absent. Paired report4a3c608bcbe728e1cb3695dcf5e5a1623ddcc79481c34c648810c6c1c0a85ef8; pair owner83fcdefecb1cda0d1a24a8a018952782ed640eb7ede677fbf31686cd4c260299. Do not repeat training.

Independent saved-pair audit COMPLETE PASS: NEW/direct_target_context_pair_fit_independent_v1/results_v1/report.json ab65dd600ac6ad8e1ff7093cc8e0a56f40285d09eca55d212124f08bd789c2af, 118097 checks, both exports qualified. Owner7f1806a2132ea3879457a3bb2705105b3075835417151cc9003143974bc99dd9. Wrapper26916/Python27048 exited0 and are absent; actual auditor executed zero model/ORT/native calls. Do not repeat audit.

CAUSAL endpoint selected for WSL witness then original full canonical simulation. Final ORT64 causal vs blinded: nominal MSE0.0003253536968 vs0.0004396891086 (-26.00%); full58 response0.0002072148232 vs0.0002175931225 (-4.77%); physical0.00008415203546 vs0.00009374958297 (-10.24%). Weighted objective improved15.37%. Counterevidence: causal full58 response remains1.0744038077 times zero-response baseline0.0001928649375. Saved improvements do not establish stability.

Root final release NEW/direct_target_causal_context_evaluation_v2/root_final_release_review.json c8002f3667de0627695262338bb5bd7a0ffeadfe85c9f003b6baa5b8d53e5dc4. Explicit causal condition, head d61915c1bf30b16431660134be6057855bbc7befeb620630a5b36dd506738f5c; checkpoint10d238198e2d7826c35eaa9bd4a13ff2e05050a7f601c1f212b585c4a90eaefd; causal owner8e6f1bfde3249bf8b8ada99a1c395c6cc3df57ce05f59ab0d91192aef466c979. All18 release roles + independent audit + owner bound. Same reviewed36 runtime files and8 helpers, namespace source/helper review949b5124b0f452a917c87660a79f294e5276a8b20bc3da4c2a516d6d88dd3a29.

Witness metadata prepared but unstarted at this update: witness_binding.json1502f0365a29c5ec8b21b2547f158a37d38999fb90260b34371e89a319e4006d (5225 pins), witness_process/launch_receipt.json f3772bc07cf1a46b4f212cafd8f13ec4ad9f1c81388b35b95e3753027982b1d8 (5230 pins). Reviewer independently checks exact concrete package; expert_resume owns ONE dispatch and completion verification after root clearance. Then canonical1569 controls + conditional250 hold, no changed strict limits. Root owns independent physics/intent replay of new trace; commands prepared in functions store causal68000PhysicsAuditCommand and causal68000IntentAuditCommand, both Started=false and not yet executed. Do not duplicate expert witness/native dispatch.

Pico agent prepares source-only causal saved-semantics auditor: original1000 features plus incoming prior23/pre-update H300, applied-target feedback, unchanged native bookkeeping. No actual audit until fresh trace and source review.

Clock correction source and saved-stage auditor reviewed. Corrected clock request NEW/independent_plant_clock_timeout_correction_v1/clock_request.json0c22640dfb617a20c31aff7356434a6253096b0fc4cbed38aae64faeb14e75eb; launch receiptf3eb4aca76e45b56761a95017090359a4de21d6edd1f3d7cacc71754158f1713, all3745 pins exact. Metadata only: no clearance, dispatch, native or model calls. Root source01939a4bf82d4f1b2e19e7240ea0f4aa4e6eba49c7dac71116a1c24b509d13c2; saved-auditor v3 root278c509418ef1c7a1be655693a840217dd8c1894a6a6c33f0757a712ad8b9ac7. Fixed setup240s, epoch+120s plant, preservation180s, GNU outer555s+5s kill. Native18190/four MJB/2ms/20ms/debt100/elapsed60s unchanged. Do not run clock concurrently with policy training/witness/canonical/independent physics.

## Preserved history

The following older status is historical, not active dispatch instructions. Completed runs and audits must not be repeated. Context study_v1 failed initial drift before any optimizer update; study_v2 resolves only first-layer GPU32 arithmetic identity using exact split1000+323 while FP64 deployment stays monolithic1323. Original clock timed out with actual native counters UNKNOWN, not zero. Full58 ordinary65000 canonical failed knee speed at control296/sub9; full evidence already audited.

'''
current.write_text(prefix + old[old.index('## Latest controller result:'):], encoding='utf-8')
session = repo / 'artifacts/teleop_resume_20260911/SESSION.md'
with session.open('a', encoding='utf-8') as f:
    f.write('\n\n' + prefix)
print(str(current))
