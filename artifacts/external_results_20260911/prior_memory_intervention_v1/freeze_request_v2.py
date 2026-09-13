"""Guard-only final revision; unexecuted v1 inputs and source remain preserved."""
import ast,json,hashlib
from pathlib import Path
BASE=Path(__file__).parent;SRC=BASE/'source_snapshot_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
r=json.loads((BASE/'request_unexecuted_v1.json').read_text())
for path in SRC.rglob('*.py'):ast.parse(path.read_text())
r['source_directory']=SRC.as_posix();r['source_sha256']={p.relative_to(SRC).as_posix():sha(p) for p in SRC.rglob('*.py')}
r['input_sha256'][(BASE/'run_durable.ps1').as_posix()]=sha(BASE/'run_durable.ps1')
r['input_sha256'][Path(__file__).as_posix()]=sha(Path(__file__))
r['revision_note']='v1 source/request/launcher preserved unexecuted; v2 adds final initial-request and initial-clearance identity guards only.'
for path,digest in r['input_sha256'].items():assert sha(Path(path))==digest,path
write(BASE/'request.json',r)
c=json.loads((BASE/'clearance_DRAFT_unexecuted_v1.json').read_text());c['request_sha256']=sha(BASE/'request.json');c['launcher_sha256']=sha(BASE/'run_durable.ps1')
write(BASE/'clearance_DRAFT.json',c)
print(json.dumps(dict(request_sha256=sha(BASE/'request.json'),launcher_sha256=sha(BASE/'run_durable.ps1'),sources=len(r['source_sha256']),inputs=len(r['input_sha256']))))
