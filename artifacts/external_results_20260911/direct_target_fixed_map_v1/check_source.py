"""Assert math unchanged under new input paths/counts and reporting names."""
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'student_physical_response_fixed_map_v1/diagnose_fixed_map.py'
NEW=BASE/'diagnose_fixed_map.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
a,b=ast.parse(OLD.read_text()),ast.parse(NEW.read_text())
for name in ('local','sha','load','read','write','exact','rms'):assert dump(fn(a,name))==dump(fn(b,name)),name
ma,mb=fn(a,'main'),fn(b,'main')
la=next(n for n in ma.body if isinstance(n,ast.For) and ast.unparse(n.target)=='control')
lb=next(n for n in mb.body if isinstance(n,ast.For) and ast.unparse(n.target)=='control')
# Through both full tangent products, nominal equality gate, clips and error:
for index,(left,right) in enumerate(zip(la.body,lb.body)):
    if isinstance(left,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='row' for t in left.targets):break
    assert dump(left)==dump(right),index
names=('tree','body','planner','scope','fake')
for name in names:
    xa=[n for n in ma.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
    xb=[n for n in mb.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
    assert [dump(n) for n in xa]==[dump(n) for n in xb],name
assert dump(next(n for n in ma.body if isinstance(n,ast.FunctionDef) and n.name=='difference'))==dump(next(n for n in mb.body if isinstance(n,ast.FunctionDef) and n.name=='difference'))
assert 'terminal_BFM_history_deviation_rms' in NEW.read_text()
r=dict(passed=True,prior_source_sha256=sha(OLD),derived_source_sha256=sha(NEW),source_checker_sha256=sha(__file__),
    preserved_math='same pure AST difference helpers and58-dimensional full K products, nominal target gate, feedback±0.1 and native clamps',
    added_reporting='explicit actual failed left ankle pitch state/command; prior/history labeled terminal handoff only',
    model_calls=0,native_steps=0,optimizer_updates=0)
with (BASE/'source_check.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r))
