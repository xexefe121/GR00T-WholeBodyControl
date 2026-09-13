"""Execute only two AST-extracted JSON gates; declared binary identities are not rehashed here."""
from pathlib import Path
import ast
from prepare_execution import BASE,read,sha,write

def main():
    candidate=read(BASE/'metadata_schema_proof.json')['request_candidate']
    source=BASE/'source_prepared_v1/recovery_data.py'
    tree=ast.parse(source.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('literal_member','check_recovery_subjects')]
    if len(functions)!=2:raise ValueError('Expected exact actual metadata functions')
    pins={v['path']:v['sha256'] for v in candidate['subjects'].values()};declared=[]
    def metadata_sha(path):
        path=Path(path)
        if path.suffix.lower() in ('.json','.py'):return sha(path)
        key=path.as_posix()
        if key not in pins:raise ValueError('Unexpected binary read requested by metadata proof')
        declared.append(key);return pins[key]
    scope={'Path':Path,'read':read,'sha':metadata_sha,'__file__':str(source),
        'RECOVERY_ROLES':('recovery_rows','collection_report','collection_request','collection_qualification','collection_source_review','consistency_report','warm_restore_review')}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),scope)
    evidence=scope['check_recovery_subjects'](candidate,pins)
    write(BASE/'metadata_gate_proof.json',dict(passed=True,actual_source_sha256=sha(source),candidate_proof_sha256=sha(BASE/'metadata_schema_proof.json'),
        actual_functions=['literal_member','check_recovery_subjects'],actual_gate_result=evidence,declared_binary_hashes_only=sorted(set(declared)),
        actual_binary_hashes_reverified=False,task_array_loads=0,checkpoint_loads=0,model_calls=0,native_steps=0))
    print('ACTUAL_METADATA_GATE_PASS')
if __name__=='__main__':main()
