"""One source-only metadata-order regression; no task imports."""
import ast
from pathlib import Path

def test_seed_certificate_and_optimized_target_are_separate_phases():
    path=Path(__file__).parent/'source_draft_v1/run_width251_actual_oracle.py'
    tree=ast.parse(path.read_text())
    functions={f.name:f for f in tree.body if isinstance(f,ast.FunctionDef)}
    def phases(node):
        return [n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='save_work']
    calls=phases(functions['certify_initial'])
    assert len(calls)==1 and ast.literal_eval(calls[0].args[0])=='initial_seed_certification_complete'
    branch=functions['run_branch']
    call=next(c for c in phases(branch) if ast.literal_eval(c.args[0])=='first_optimized_target_ready')
    guard=next(n for n in ast.walk(branch) if isinstance(n,ast.If) and call in list(ast.walk(n)) and
               ast.unparse(n.test)=='control == START')
    assert len(guard.body)==2
    assert isinstance(guard.body[0].value,ast.Call) and guard.body[0].value.func.id=='atomic_trace'
    assert guard.body[1].value is call
    source=path.read_text()
    assert 'first_query_complete' not in source
