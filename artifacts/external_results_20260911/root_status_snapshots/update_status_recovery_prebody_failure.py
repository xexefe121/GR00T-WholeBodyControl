from pathlib import Path
from datetime import datetime,timezone
import json
new=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911');art=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911')
now=datetime.now(timezone.utc).isoformat()
note='''Recovery first LAUNCH failed before target Linux script/Python. ONE21:50:47→21:50:54UTC wrapper23604/WSL14824, raw127/diagnostic1/all77postpins exact. stderr /bin/bash: line1: -d: command not found. Quoting every fixed WSL option made WSL interpret -d as a shell command. No linux_process.json, ATTEMPT_STARTED or work ledger; no selected recovery model/native/replan work entered. Expert soleowner preserving this prebody attempt and absence; no automatic retry.

Main20/request/frozen/selected actual pre251 boundary remain unchanged. Root authorized ONLY corrected launcher/process_v2/owner source and a real non-numerical Start-Process+WSL argv/mount-bootstrap probe. Reviewer pauses conditional collection to review this correction. Need final amended concrete root review before dispatch. No actual recovery query has run. Preserve first launch/source/clearance/process records.

First requestc3e20cd213091bd9ec0db5a99871404ae609712de5be3e664b6bd844f62edea8; frozen954ac6190406c575964db094fdd69d9fb387482a5ba2f19ed289b2194f698ce1; launch7012ee94d666085f2c48256949a94f34392ccac6388e2ee928248c383ad536c0; launcherf89670e1bd4749ae7b8f85a3d124235b5c2259a49d6e7e6f05d8f39945bcb836; rootconcrete7ada54d355e90aa7299bf0339582ece7cbb6feee617c57eda6fb74593f1b7489; independentconcretee6900a1101f38d8e50839cfee81a2b2aaf72c0534e7a3309fa6dd3603cfed1a4; clearance29d674c341ae0264a44a2189a15740a88b3a5e218423173e4fa59743167d6260. Boundary367checks20bb1e718966c902c121e0812b1eb4f0ea28dea8601ae5cf172425c32247d06e; snapshotaa8cd94cdefc77b96624c75b4825a1133d805b8fd8be0717dc765e4c8f00c3c3; prefix8a755a3bf1acd751c456f04805017f8dd70a025988e61c11e4690a07d7f48649; selection1c71c7c8d69b491a56e27b3e2ca8854e6287c160473c41a5f33fc28c8d0d2596. Previous helper-only original68pin launch preserved, final74pin owner improved strict identity and interrupted-work accounting; all actual main/source unchanged.

Pico source-only timing probe v2 preparation4c0b40ae3decb12af8165b52fdbfb578d3edc946b2d286cc3112fc7fb2d6af19,31fakes; probea41ef15e1e47afc5d6fb3c0ab27a203208b1c66c10559a1ae8792e6e0dbe57f4. Root found v1 global per-read monotonic thresholds could falsely reject legitimate nested GC interruptions; v1 preserved, v2 uses local/pair/root chronology. Integration source-only underway; no next clock selected. Root compare_first_target.py +3fakes ready, no actual comparison. Conditional collector source-only controls251..1268, no labels or fit.'''
path=art/'CURRENT.md';s=path.read_text(encoding='utf-8-sig')
with (new/'root_status_snapshots/CURRENT_before_recovery_prebody_failure.md').open('x',encoding='utf-8') as f:f.write(s)
i=s.index('\n## Latest completed audit')
s=s[:i]+'\n## Latest recovery launch state\n\n'+note+'\n'+s[i:]
lines=s.splitlines();lines[2]='Updated '+now+'. User said continue. Fast full-body simulation unqualified; real Pico/DDS/robot waits. Latest section supersedes older preparation statements below.'
path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
with (art/'SESSION.md').open('a',encoding='utf-8') as f:f.write('\n\n'+now+': '+note+'\n')
print(json.dumps({'updated_utc':now,'prebody_launch_failed':True,'actual_recovery_query_started':False,'corrected_launcher_pending_review':True}))
