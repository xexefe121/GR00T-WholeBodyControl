"""Preserve accumulated status, publish a concise current handoff."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
NEW=Path(__file__).resolve().parent
DOC=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911')
now=datetime.now(timezone.utc)
backup=NEW/'root_status_snapshots'/('before_recovery_path_fix_'+now.strftime('%Y%m%dT%H%M%SZ'))
backup.mkdir(parents=True,exist_ok=False)
for name in ('CURRENT.md','SIM_RESULT.md'):
 p=DOC/name
 if p.exists():(backup/name).write_bytes(p.read_bytes())
body=f'''# Current native23 simulation work

Updated {now.isoformat()}. User requests continued work. Full 23-DOF real-time teleoperation remains unqualified. Real Pico/DDS/robot waits for simulation confidence.

## Active work

Fresh expert recovery from actual width81000 pre-control251 is selected. Exact actual full291 state at t5.02, incoming300 history, prior applied-target inverse23, and all251 executed prefix controls remain fixed. Intended continuation:1018 MPC controls/204 solves,300 terminal controls, then250 continuous hold;15,680 new native steps. No fresh expert plan has run yet.

Expert owns metadata-only replacement in **direct_target_width512_expert_recovery_v2**. Preserve v1; copy same20 source files and already-extracted input bytes. Normalize all frozen input path keys and matching request subjects to forward slashes. Validate every mapped frozen file with actual read-only WSL hashes before one separately recorded launch. Reviewer checks this narrow derivative; root then selects corrected concrete invocation. No repeat training, canonical trial, or input extraction.

First v1 transport launch21:50:47 failed before Python: quoted WSL options caused raw127. Corrected transport launched22:08:08, wrapper24156/WSL17704/Linux373, reached Python and failed22:08:21 in inherited frozen-input gate: drive converted but Windows backslashes retained. Final ledger all21 work categories zero; no Batch/native/model/replan. All106 pins exact and all processes absent. Completed owner_v2 a0eaedf23819e9187c79ea0640905f16cc89a250071c3bfc3c25192799d6902d has accounting PASS, recovery FALSE. Both attempts preserved; no numerical result inferred.

Root transport review v2 110e578aa2affa04a02a1e5b6d3a678bbd5a1843b62990db68d65db3b52602c5; launch29484b217185eab4e5508dd2b524176fb56b21f3f601dcbf415bb62bb13e8fcc; new clearancea4816dad373dc98db0bd6e76bb0600833cfec21f09e14875dc34486ad1d87f41. Original canonical clearance29d674c341ae0264a44a2189a15740a88b3a5e218423173e4fa59743167d6260 unchanged.

Pico prepares timing sidecar auditor source only. Integrated probe source32 files/20 of24 old unchanged passed root review fdccbf0e31e716190a805cd2d5e6f52d3aae629e18136386f4da9d1e95b78310; independent150 pytest cases+20 subtests passed. It measures wall/thread/process CPU and GC overlap by stage, preserves original deadlines/control/physics, and keeps partial native ownership on hook faults. No actual instrumented clock selected or run. Finish sidecar auditor before selecting that trial.

Root owns fresh first-target saved comparison and full/hold original physics+intent audits when actual recovery outputs permit. Prepared pure comparison helper has3 passing synthetic cases; no actual comparison. Conditional collector preparation only; exactly controls251..1268 after owner and four audit passes. Connected expert states are not independent queries of later student states. No next fit selected.

## Latest completed evidence — do not repeat

Width81000 larger512 head: fit completed10,000 updates, output parity passed. Canonical trial failed control309/sub6 at right-hip-pitch speed32.112498 versus32rad/s;3096 native steps,60 learned controls,zero source controls. Root physics reproduced all3096 steps/seven fields exactly; intent failed. Saved semantics7193 checks exact; first departure251, first clipping264. Better training error did not establish balance. All relevant reports/owners remain under direct_target_causal_width512_student_v1, direct_target_causal_width512_evaluation_v1, and direct_target_width512_* review directories.

Pending-result clock trial failed after12663 native steps/12662 verified. Tick12099 took23.076358ms, next job1211 created1.414741ms after immutable deadline;40 misses,56 held controls, then joint bound. Worker BUSY retry was not exercised. Matching saved audit14659b36321c3f5b9ef825a9029e95a2388ea65c45ec4f3f24802659e8e8157a passed22,963 comparisons; owner66a2af16af7a3e861122b793001371b908bdf98584c937e41d341db43091a0f9. Physics/timing/command qualifications remain failed. Stall cause currently unknown.

Slow offline expert traces previously passed full PICO, walk002, walk003 plus continuous holds; walk008 source passed slow expert. Same fast controller has not passed complete suite. Original SIM_ACCEPTANCE.md and all native limits, reference timing and source windows remain unchanged.

## Preserved detailed history

Earlier complete status, hashes, completed experiments and no-repeat ledger: [{backup.joinpath('CURRENT.md').name}]({backup.joinpath('CURRENT.md').as_posix()}). SESSION.md retains append-only progress. Large artifacts stay on E:.
'''
(DOC/'CURRENT.md').write_text(body,encoding='utf-8')
with (DOC/'SESSION.md').open('a',encoding='utf-8') as f:
 f.write('\n\n'+body)
print(json.dumps(dict(updated=str(DOC/'CURRENT.md'),prior_snapshot=str(backup),sha256=hashlib.sha256(body.encode()).hexdigest())))
