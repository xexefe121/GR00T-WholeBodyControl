"""Static proof of original semantics under explicitly declared hook erasure.

Never imports producer/native/task modules. Erases only listed instrumentation
forms, then requires the complete three modules' original ASTs to match.
"""
import ast
import copy
import hashlib
from pathlib import Path

MODULES=('clock_core.py','native_stepper.py','session.py')

def dump(node):return ast.dump(node,include_attributes=False)

class EraseHooks(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        if node.module=='timing_hooks':
            assert [(v.name,v.asname) for v in node.names]==[('TimingHooks',None)]
            return None
        return node
    def visit_With(self,node):
        node=self.generic_visit(node)
        if len(node.items)==1:
            call=node.items[0].context_expr
            if isinstance(call,ast.Call) and ast.unparse(call.func) in ('self.timing.span','self.timing.native_span'):
                assert node.items[0].optional_vars is None
                return node.body
        return node
    def visit_Assign(self,node):
        if len(node.targets)==1 and ast.unparse(node.targets[0]) in ('self.timing','self.foundation.timing','self.stepper.timing'):
            assert ast.unparse(node.value) in ('TimingHooks()','TimingHooks() if timing is None else timing','self.timing')
            return None
        return node
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='self.timing.require_before_native':
            assert not node.value.args and not node.value.keywords
            return None
        return node
    def visit_If(self,node):
        if ast.unparse(node.test)=='self.timing.failed':
            expected=ast.parse("if self.timing.failed:\n self.driver_failure=encode(dict(type='TimingInstrumentationFailure',detail=self.timing.status(),phase='before-next-tick'))\n return False").body[0]
            assert dump(node)==dump(expected)
            return None
        return self.generic_visit(node)
    def visit_FunctionDef(self,node):
        for i,arg in enumerate(node.args.kwonlyargs):
            if arg.arg=='timing':
                assert node.name=='__init__' and i==len(node.args.kwonlyargs)-1
                assert dump(node.args.kw_defaults[i])==dump(ast.Constant(None))
                del node.args.kwonlyargs[i];del node.args.kw_defaults[i]
                break
        return self.generic_visit(node)


def check(original,integrated):
    result={}
    for name in MODULES:
        old=ast.parse((original/name).read_text());new=ast.parse((integrated/name).read_text())
        erased=EraseHooks().visit(copy.deepcopy(new))
        assert dump(old)==dump(erased),'undeclared semantic delta: '+name
        result[name]={'whole_module_after_declared_hook_erasure_exact':True,
            'original_sha256':hashlib.sha256((original/name).read_bytes()).hexdigest(),
            'integrated_sha256':hashlib.sha256((integrated/name).read_bytes()).hexdigest()}
    before=ast.parse((original/'run_clock.py').read_text());after=ast.parse((integrated/'run_clock.py').read_text())
    old=[n for n in ast.walk(before) if isinstance(n,ast.While)]
    new=[n for n in ast.walk(after) if isinstance(n,ast.While)]
    assert len(old)==len(new)==1 and dump(old[0])==dump(new[0]),'outer fixed loop changed'
    result['outer_hot_while_ast_exact']=True
    return result


if __name__=='__main__':
    import json
    base=Path(__file__).resolve().parent.parent
    print(json.dumps(check(base/'source_original_v1',base/'source_draft_v1'),indent=2))
