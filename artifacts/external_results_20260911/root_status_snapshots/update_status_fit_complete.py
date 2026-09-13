from pathlib import Path
from datetime import datetime,timezone
NEW=Path(__file__).resolve().parent.parent;OUT=NEW/'root_status_snapshots'
ART=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911')
stamp=datetime.now(timezone.utc).isoformat()
for name in ('CURRENT.md','SIM_RESULT.md'):
    with (OUT/(name.replace('.md','')+'_before_fit_complete.md')).open('x',encoding='utf-8') as f:f.write((ART/name).read_text(encoding='utf-8-sig'))
s=(ART/'CURRENT.md').read_text(encoding='utf-8-sig')
start=s.index('Corrected ONE warm balanced fit v2');end=s.index('Protocol unchanged:',start)
s=s[:start]+'''Corrected ONE warm balanced fit v2 COMPLETE; do not repeat. Report NEW/direct_target_causal_response_balanced_student_v2/fit/report.json f918f1dfe675aae01e25307d302ce6c54f599fb7f0f5d1fa3fd943a179b5aded; owner a6b75e9df7864e46a5ed0ca84efe05fc0fa2e020b25766d6077b3e6bc744b8d3. PT395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d, ONNXc70e90adab31efd2abcebd9817bf3c3c1ecc84cde546e080f6367c710633423a. All3000updates/fivebackends complete, raw/wrapper0, PIDs24996+25004 absent, allpins; worst export gate1.0272860784255045e-8rad. Initial predictions all367570 byteexact to causal68000. GPU32 initial-to-final N-3.520%, originalF-2.341%, balancedF-8.951%, P-4.636%, balancedtotal-6.641%; finalORT F/zero1.049253, balancedF/zero1.330734. Numerical completion does not establish stability.

'''+s[end:]
start=s.index('Reviewer prepares independent single71000');end=s.index('Evaluator prepared under',start)
s=s[:start]+'''First saved-fit audit COMPLETE FAILED, preserved. NEW/direct_target_response_balanced_fit_independent_v1/results_v1/report.json e6013ad7597b772915cfd3fea6b04ccd175082d96a5516f64abb2a7bb7e40576;1948checks then checkpoint_weight_rule. Raw/wrapper1, PIDs17260+27920 absent, no model/ORT/gradient/native calls. Root concrete8b790461 bound requestf23b1c47/launch4f007153/all36pins. Diagnosis NEW/direct_target_response_fit_audit_rule_diagnosis_v1/report.json d912a020b2a7c9e999c9165b1f5ed1749c3309e3736caf533d993799d98605dc confirms auditor wrongly reused energy rule text Emean/Eg for checkpoint/report/metrics; producer exact rule is mean_six_teacher_group_energies_over_group_energy. Six numeric weights match; energy_source rule correctly remains Emean/Eg. Reviewer prepares fresh v2 source/regressions; no new audit selected, no training repeated. Root final-release writer prepared but not run.

'''+s[end:]
s=s.replace('Pico prepares separate retry-aware saved-protocol audit;','Pico prepares retry-aware saved-protocol auditor v2 after review found expiry could skip tick coverage and StageTimeout BaseException was not recognized as terminal interruption; v1 remains preserved. No producer change. Old auditor core checks and fixed deadlines remain. ')
(ART/'CURRENT.md').write_text(s,encoding='utf-8')
r=(ART/'SIM_RESULT.md').read_text(encoding='utf-8-sig');start=r.index('## Work now underway');end=r.index('## Existing qualified offline evidence',start)
r=r[:start]+'''## Work now underway

Fixed continuation to ordinary71000 completed: all3000updates, five full numerical backends and export checks passed. Worst same-weight export error1.03e-8rad. Balanced response objective fell8.95%, but remains1.33times zero-response baseline; controller stability unqualified. Initial predictions matched previous checkpoint exactly. No checkpoint selection or extra training.

First independent saved-fit audit stopped on an auditor metadata error after1948checks: it expected abbreviated weight-rule text where the producer stores the pinned full rule. Failed audit preserved; checkpoint/report/metrics and numeric weights remain unchanged. Fresh auditor correction under review before any simulation launch.

Recorded-command clock benchmark remains failed: command420 hit BUSY19.464ms before its deadline and was never delivered; six2ms timing misses occurred, first before the dropped command. Completed saved audit passed8605 integrity checks. Bounded same-payload retry source passed67tests plus five subtests. Retry-aware audit review found terminal expiry accounting gaps; fixing those before another timed run. Physical, command and timing qualification remain false.

'''+r[end:]
(ART/'SIM_RESULT.md').write_text(r,encoding='utf-8')
with (ART/'SESSION.md').open('a',encoding='utf-8') as f:f.write('\n\n'+stamp+': ordinary71000 fit COMPLETE numerical PASS, reportf918f1df/ownera6b75e9d; saved-fit audit v1 failed1948checks on auditor weight-rule string, actualfit unchanged. Fresh auditv2 being prepared. No canonical run selected. Retry-aware clock auditv2 fixes expiry/terminal evidence before new run.\n')
print('Status updated; prior versions preserved.')
