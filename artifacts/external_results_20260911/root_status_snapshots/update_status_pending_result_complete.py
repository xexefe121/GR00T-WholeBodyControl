from pathlib import Path
from datetime import datetime, timezone
import json

new = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
art = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911')
now = datetime.now(timezone.utc).isoformat()
summary = '''Pending-result clock COMPLETE FAILED; do not repeat. ONE wrapper5620/child2840 and Linux388/469/470 all absent; raw/diagnostic/wrapper2, no watchdog timeout, all3782 pins exact. Report209b04cc616fce8b51ba76d7af593533f1099427da33c4b30fa90db0b8da3bff; owner968a2e8bb369a9168c0383503cb4775c1bcf87bf5482703d8c233ca3476017ae. Exactly12663 native attempted/returned/captured,12662 verified, native_joint_bound failure. Command1211 missed; controls1211..1266 held old command. Worker EXPIRED_BEFORE_REPLY;40 foundation/outer deadline misses. Allfour MJB717cc6f0 exact. No numerical trial active. Pico owns matching saved-audit preparation and saved diagnosis only; actual audit still awaits concrete root review. Clock requestbfa3e9364ca648c810260b39ee86d4480a20a69340c399978338871f655d1f3e; launch8e9ef98e4bfcab9f8f7c3eaa9d6dda4ac3f6b3eb7151734d5dfb61398c0111b2; rootconcreted2bb6dcebcffe09a71160b156365092fd96936868e7043ce7a09f6855a90992e; clearance64d2184e67389cb9a01d5fbf3e478b5956a4a30527659a95c5d460db2b6131f4.

Recovery remains source-only: one prespecified actual width precontrol251, full291 plus incoming323 causal context; actual prefix251 includes learned250. Exact original1318 remaining controls and conditional250hold. Separate private BFM61200, initial-certificate900, restoration-certificate122400, preview15680 native-step ceilings, aggregate200180; actual15680. FD150552000 and line20933100 lane-step ceilings. Expert finishing final tests/freeze; independent reviewer checking hooks, budgets and prefix. No actual replan, labels or next fit selected.'''
path = art/'CURRENT.md'
s = path.read_text(encoding='utf-8-sig')
with (new/'root_status_snapshots/CURRENT_before_pending_result_complete.md').open('x', encoding='utf-8') as stream:
    stream.write(s)
start = s.index('## Clock result retry source ready; no new clock run')
end = s.index('\n\nPrior pending-publication clock', start)
s = s[:start] + '## Pending-result clock complete; saved audit next\n\n' + summary + s[end:]
s = s.replace('No model/native work active now. Agents may resume synthetic tests; no new clock or fit selected.', 'No model/native work active now. Pending-result clock also completed below; fresh expert recovery remains source preparation only.')
s = s.replace('Reviewer now preparing width81000 saved-semantics source only; no actual trace assumed.', 'Width81000 saved semantics subsequently completed7193checks, as recorded above.')
lines = s.splitlines()
lines[2] = 'Updated '+now+'. User said continue. Fast full-body simulation remains unqualified; real Pico/DDS/robot waits. NEW=E:/codex-artifacts/sonic23_teleop_resume_20260911. Completed runs and failures preserved.'
path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
with (art/'SESSION.md').open('a', encoding='utf-8') as stream:
    stream.write('\n\n'+now+': '+summary+'\n')
path = art/'SIM_RESULT.md'
s = path.read_text(encoding='utf-8-sig')
with (new/'root_status_snapshots/SIM_RESULT_before_pending_result_complete.md').open('x', encoding='utf-8') as stream:
    stream.write(s)
path.write_text('# Latest simulation result\n\n'+now+'\n\nFast full-body controller remains unqualified. Width81000 failed at control309, step3096; independent physics reproduced every recorded step exactly. Saved history audit passed7193checks, first departure251 and clipping264.\n\n'+summary+'\n\nEarlier result, retained for provenance:\n\n'+s, encoding='utf-8')
print(json.dumps({'updated_utc': now, 'clock_complete': True, 'recovery_source_only': True, 'actual_calls': 0}))
