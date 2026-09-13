"""Validate unchanged durable helper functions; source-only, no stage creation."""
import ast
from derive_sources import BASE,OLD
def main():
    def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert functions(BASE/'prepare_audit_stage.py')==functions(OLD/'prepare_audit_stage.py')
    assert (BASE/'verify_completion.py').read_bytes()==(OLD/'verify_completion.py').read_bytes()
    assert (BASE/'preserved_run_audit_template.ps1.txt').read_bytes()==(OLD/'preserved_run_audit_template.ps1.txt').read_bytes()
if __name__=='__main__':main()
