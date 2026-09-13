from pathlib import Path
from datetime import datetime,timezone
NEW=Path(__file__).resolve().parent.parent;OUT=NEW/'root_status_snapshots'
ART=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911')
stamp=datetime.now(timezone.utc).isoformat()
for name in ('CURRENT.md','SIM_RESULT.md'):
    with (OUT/(name.replace('.md','')+'_before_balanced_repair_v2.md')).open('x',encoding='utf-8') as f:f.write((ART/name).read_text(encoding='utf-8-sig'))
s=(ART/'CURRENT.md').read_text(encoding='utf-8-sig');start=s.index('## Active work');end=s.index('## Latest controller:',start)
active=f'''## Active work ({stamp})

Corrected ONE warm balanced fit v2 dispatched18:55:12UTC, wrapper24996; expert_resume owns all monitoring/owner verification. Do not duplicate. NEW/direct_target_causal_response_balanced_student_v2/training_request.json366c31139a99234d710037642b9f26124620ff6190501c41f293546a648c074e; frozen e1423363c6af8e6e88c04f3cfaf507d29416167b1bafe81d408ddcd5088dba13; launcherf5dc300610b242c6026163b652cd0304c585fc71a0e42502311510b49b5c744a; clearancebbe70dc81dd7142b080d1a4c3fb41f047000c9d1d1b98878243043adc3fceb4c. Root concrete d4958e16decbf4a9209b006d91e3c8a47c7a2f9fe0f451bb757b2bcd52c5e3b9 checks351inputs/25sources. Source403cea76732f52ccc0aa8e81da393e517a87e549c96e7be4ceb31a612550065a; rootrepair935b27c9f93fa21f31552dd0707174076f490d26a18f00dfd28eb9eb3ff03280.24sources byte-exact to v1; only carried-request writer uses dict(request,condition=...) to avoid duplicate keyword.

Protocol unchanged: source causal68000 PT10d238198e2d7826c35eaa9bd4a13ff2e05050a7f601c1f212b585c4a90eaefd; exact learned1323 actor, AdamW moments/step3000, RNG/norm, saved contexts and schedules;3000updates/9000trainforwards/44,058,000rows; endpoint71000/optimizer6000; inclusive LR1e-5→1e-6; fixed coefficient1.8188207859141674 and six fixed teacher-energy response weights; N/P unchanged. Original initial/final1e-5rad gates, byte equality diagnostic only, allfive backends/original+balanced54cells. No adaptive retry or best-checkpoint selection. One engineering continuation does not isolate weighting from extra epochs/optimizer choices.

Failed v1 is COMPLETE PRESERVED and must never rerun. Wrapper24400/child17036 raw/wrapper1, PIDsabsent, allpins; failure duplicate condition before first forward/update. Ownerf49361bcb6b0bee4160569b611c30dbc6f28c5597917caed39d2c450c8d82827; producer savedzero-call checkfdfef59bf5f5517ed1e67b3ace724e1d438238b7f9a9c9b6c84672fdbdf785fd; independentroot NEW/direct_target_response_initial_failure_root_v1/report.json50c29d574ec3b17375a81b468da443ba6d80b759e257b51906ba054ed329e646. Source/init/failed actor+moments+RNG/norm exact; all six steps3000, all training/diagnostic/native/gradient counts0. Three root and producer actual-expression regression tests prove narrow fix.

Reviewer prepares independent single71000 saved-fit auditor under NEW/direct_target_response_balanced_fit_independent_v1; no actual audit selected. Root reviewed draft math/full driver and corrected top-level concrete-review field assumptions and matching carried-request construction.39synthetic tests reported; final source receipt pending.

Evaluator prepared under NEW/direct_target_causal_response_evaluation_v1:37sources,33 old runtime files byte-identical, only single warm71000 release gates/test plus reviewed balance constants. Source2195d4f5d06b6c4dec5d2a24038a179be0e266483fc0454edea540705d40fef3; independent source review e40aca20c419219a8e6d1e4630ef5e4442ac7c1f492d85b009bc94b1460a0121; root and independent83tests PASS.16 literal endpoint/data/energy roles. Expert adapts eight helpers to fitv2 with old helper version preserved. No witness/canonical binding or actual controller selected until saved-fit audit. Root functions causal71000PhysicsAuditCommand and causal71000IntentAuditCommand prepared, Started=false; not executed. Corrected intent includes repo PYTHONPATH.

Clock saved audit COMPLETE PASS (evidence only): NEW/independent_plant_clock_saved_actual_v1/results_v1/report.json48e1a6275a70998da3a2dfdda6f017b5a488c8ff4bd422d3329b4f98267ec8b1; owner41bf784a6423e4c220f79f87ca465205c2b0be61df5b78870a8b4f16f8d97e1f.8605checks, all3802pins/raw0/wrapper0/PIDs23472+25812absent; actual model/nativecalls0. Physical/timing/command/component false. Do not repeat. Saved-byte BUSY diagnosis receiptbbbb8a9d38554ead77a32b9d224647775c517dac518125ebbad68837b6f18975;6422envelopes/419full chains exact. Worker lock interval omitted from original EMPTY-poll ledger, so do not invent its timing.

Bounded pending-publication source NEW/independent_plant_pending_publication_v1/source_preparation.json6cb9f8bf38543634c32e745496f63fe94e23391fd10948be41749eaea5356575; rootreview NEW/independent_pending_publication_root_review_v1/review.jsond2f303a488817a1f088790fbe37a8d411576ddfb7b84c1043c5309dbc7a4e357.22sources/20oldexact, onlyclock_core changed plus tests; root67+5subtests PASS. Same immutable job; initial+9nonblocking attempts, only BUSY, once/tick before original activation. Retry guard and actual transport times separate; late replies still rejected. Original native/2ms/20ms/debt/elapsed/watchdog/80kledgers unchanged. Pico prepares separate retry-aware saved-protocol audit; old v3 explicitly cannot audit new repeated publication/expiry ledger. No new clock request/launch selected; never run clock concurrently with heavy training/physics/fit-audit work.

'''
s=s[:start]+active+s[end:]
(ART/'CURRENT.md').write_text(s,encoding='utf-8')
r=(ART/'SIM_RESULT.md').read_text(encoding='utf-8-sig');start=r.index('## Work now underway');end=r.index('## Existing qualified offline evidence',start)
r=r[:start]+'''## Work now underway

Corrected fixed3000-update continuation toordinary71000 launched at18:55UTC. It restores causal68000 actor, full AdamW moments/step3000, RNG, normalization and saved context/schedule. Original nominal/physical objectives and coefficient stay fixed; each response group is weighted by mean teacher energy divided by that group's energy. All original and balanced metrics remain reported. The first launch failed before any forward pass or optimizer update because the saved-request writer supplied condition twice. Actor, optimizer and RNG remained exact; failure and process evidence are preserved. A fresh directory fixes only that writer line, with three regression tests. No result from the corrected fit is claimed yet.

The unchanged native evaluator is prepared with71000 release checks and passed83 independent synthetic tests. It requires complete saved-fit and export checks before any controller test. Independent saved-fit audit preparation runs alongside training.

Corrected recorded-command clock benchmark returned4694 native steps and four exact model snapshots, then failed a native joint bound. Command420 hit a BUSY mailbox19.464ms before its deadline and was never retried or delivered. The plant held419 through control469. Six2ms timing misses also occurred, first at step610 before the command failure. The completed saved-only audit passed8605 integrity checks; physical, command and timing qualification remain false.

A bounded nonblocking publication retry now passes67 independent tests plus five subtests. It preserves command bytes, original deadline and late-result rejection; scheduler stalls remain measured. A separate saved-protocol auditor is being adapted for retry evidence. No new clock benchmark is selected, and correcting BUSY alone cannot establish2ms timing.

'''+r[end:]
(ART/'SIM_RESULT.md').write_text(r,encoding='utf-8')
with (ART/'SESSION.md').open('a',encoding='utf-8') as f:f.write('\n\n'+stamp+': corrected balanced fitv2 dispatched18:55:12 wrapper24996, expert soleowner; request366c3113/clearancebbe70dc8. v1zero-forward duplicatekeywordfailure fullypreserved/audited. Clock saved integrity8605PASS with actualcomponentfalse. Retrysource rootd2f303a4passed; no newclockselected. See CURRENT.md for exact identities.\n')
print('Current status updated; previous versions preserved.')
