"""Record source-only independent paired-audit preparation."""
from pathlib import Path
import ast,hashlib,json,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'source_draft_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources={p.name:sha(p) for p in sorted(SOURCE.glob('*.py'))}
for p in SOURCE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
old=BASE.parent/'direct_target_full_state_fit_independent_v1/source_draft_v1'
unchanged=['audit_math.py','audit_full_state_math.py','audit_restoration.py']
assert all(sha(old/n)==sources[n] for n in unchanged)
assert (SOURCE/'audit_graph.py').read_text()==(old/'audit_graph.py').read_text().replace('1000','1323')
tree=ET.parse(BASE/'tests_v2.xml').getroot();suites=tree.findall('testsuite')
assert sum(int(s.get('tests','0')) for s in suites)==16
assert all(int(s.get(k,'0'))==0 for s in suites for k in ('errors','failures','skipped'))
helpers={name:sha(BASE/name) for name in ('prepare_audit_request.py','freeze_launch.py','run_audit_durable.ps1','verify_completion.py','record_preparation.py')}
result=dict(source_preparation_passed=True,preparation_only=True,source_directory=SOURCE.as_posix(),source_sha256=sources,helper_sha256=helpers,
 unchanged_qualified_modules=unchanged,graph_derivation='Only1000 to1323 shape literals changed; entire remaining qualified graph checker text unchanged.',
 reused_scope='Qualified full58 metrics, PCG64 sequence, saved-tree comparison and exact promoted20-node graph; no producer imports.',
 new_scope=['All9899 nominal context transitions and all3054 physical applied-prior/advanced-history rows; held center context and original1000 identity.',
 'Independent equal15-cell context moments, fresh same65000 expanded actor/AdamW/RNG, zero unused blinded columns and optimizer moments, same3000 schedule and lower cosine.',
 'Both3000 loss/cell ledgers, all10 complete backend passes,14370 exact diagnostic partitions,18 parity and18 drift arrays with clipping, final cell/odd/even/zero metrics.',
 'Literal actual request/frozen/review/runtime/manifest/process/owner-v2 chain, both18-role condition receipts, boolean export_qualified and separate condition map. No endpoint selected by auditor.'],
 source_tests_passed=16,tests=dict(path=(BASE/'tests_v2.xml').as_posix(),sha256=sha(BASE/'tests_v2.xml')),
 preserved_initial_test_attempt=dict(path=(BASE/'tests_v1.xml').as_posix(),sha256=sha(BASE/'tests_v1.xml'),reason='pytest traversed unrelated E:/WpSystem while finding a collection root; no tests collected. Explicit local rootdir fixed invocation.'),
 limitation='Actual request preparation requires a completed pair and positive owner accounting. An incomplete producer prefix is retained by its owner and cannot receive paired release; this auditor fails closed and retains its own checks/report on any mismatch.',
 actual_audit_request_present=(BASE/'audit_request.json').exists(),actual_audit_run=False,task_model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0)
assert not result['actual_audit_request_present']
with (BASE/'source_preparation.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(source_preparation_sha256=sha(BASE/'source_preparation.json'),sources=len(sources),helpers=len(helpers),tests=16)))
