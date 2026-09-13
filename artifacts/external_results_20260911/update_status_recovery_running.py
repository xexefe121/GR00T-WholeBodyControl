from pathlib import Path
from datetime import datetime,timezone
import json
NEW=Path(__file__).resolve().parent
DOC=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911')
path=DOC/'CURRENT.md';body=path.read_text(encoding='utf-8')
old='No fresh expert plan has run yet.'
assert old in body
body=body.replace(old,'Fresh recovery_v2 is now RUNNING. Initial full291/history restoration and H30 seed feasibility passed; first optimized target saved. Final completion and independent audits remain pending.')
old='Expert owns metadata-only replacement in **direct_target_width512_expert_recovery_v2**.'
assert old in body
body=body.replace(old,'Expert owns active actual run in **direct_target_width512_expert_recovery_v2**. Metadata repair completed; all44 inherited frozen input paths and20 source hashes passed actual WSL read-only verification. Root concrete review1282bf0fc84abe18290b7bab0469506fae5e4fe4ac8987732cc631353a11866f selected ONE dispatch22:16:49.270UTC, wrapper13580/WSL28320/Linux371, canonical clearancedb2763e9ac89b2a1041500e7878721ab74eaea308c4a8fcebdb62f1f525d458d. The following preparation requirements are now fulfilled:')
body=body.replace('Root owns fresh first-target saved comparison','Root owns fresh first-target saved comparison')
path.write_text(body,encoding='utf-8')
with (DOC/'SESSION.md').open('a',encoding='utf-8') as f:f.write('\n\n'+datetime.now(timezone.utc).isoformat()+' Recovery_v2 active: one dispatch22:16:49.270, wrapper13580/WSL28320/Linux371. All44 actual inherited WSL paths+20 sources exact; root1282bf0f; clearancedb2763e9. Full291/history restoration and initial H30 seed pass; first optimized target saved. No final completion/physics/intent/label admission yet. Root first-target comparator CLI prepared, not run.\n')
print(json.dumps(dict(updated=str(path),actual_recovery_running=True)))
