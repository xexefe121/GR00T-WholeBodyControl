"""Bind additive source/graph preparation after synthetic tests; no model imports."""
import ast,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
BASE=Path(__file__).resolve().parent;NEW=BASE.parent;SOURCE=BASE/'source_draft_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')

def main():
    derivation=read(BASE/'derivation.json');original=Path(derivation['original']['path'])
    assert sha(original)==derivation['original']['sha256']
    current=SOURCE/'width512_promoted.py';assert sha(current)==derivation['derived']['sha256']
    text=original.read_text()
    for a,b in derivation['exact_replacements']:
        assert text.count(a)==1;text=text.replace(a,b)
    assert text==current.read_text()
    old_tree,new_tree=ast.parse(original.read_text()),ast.parse(current.read_text())
    old_class=next(v for v in old_tree.body if isinstance(v,ast.ClassDef));new_class=next(v for v in new_tree.body if isinstance(v,ast.ClassDef))
    assert ast.dump(old_class,include_attributes=False)==ast.dump(new_class,include_attributes=False)
    # Exporter differs only in descriptive graph name; all operations/casts stay exact.
    old_export=next(v for v in old_tree.body if isinstance(v,ast.FunctionDef) and v.name=='export_onnx')
    new_export=next(v for v in new_tree.body if isinstance(v,ast.FunctionDef) and v.name=='export_onnx')
    for node in ast.walk(new_export):
        if isinstance(node,ast.Constant) and node.value=='same_81000_width512_context_weights_fp64_execution':
            node.value='same_71000_context_weights_fp64_execution'
    assert ast.dump(old_export,include_attributes=False)==ast.dump(new_export,include_attributes=False)
    runtime=read(BASE/'runtime.json');end=read(BASE/'test_exit.json')
    assert runtime['torch_threads']==runtime['interop_threads']==1 and runtime['cuda_initialized'] is False
    assert end['exit_code']==0 and end['cuda_initialized'] is False
    assert all(end[k]==0 for k in ('Torch_forward_calls','ORT_sessions','ORT_calls','native_steps','optimizer_updates','actual_checkpoint_loads','task_arrays_loaded'))
    xml=ET.parse(BASE/'synthetic_tests.xml').getroot();suites=[xml] if xml.tag=='testsuite' else list(xml)
    count=sum(int(s.attrib['tests']) for s in suites)
    assert count==20 and all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
    source_map={p.name:sha(p) for p in SOURCE.iterdir() if p.is_file()}
    inputs={original.as_posix():sha(original)}
    for p in [BASE/'derive_source.py',BASE/'derivation.json',BASE/'source_changes.diff',BASE/'run_synthetics.py',Path(__file__),
        BASE/'runtime.json',BASE/'test_exit.json',BASE/'synthetic_tests.xml',BASE/'synthetic_tests.log',BASE/'SCHEMA_before_gate_clarification.md',
        NEW/'direct_target_causal_width512_preparation_v1/proposal.json',
        NEW/'direct_target_causal_width512_preparation_v1/source_preparation.json',
        NEW/'direct_target_width512_independent_source_review_v1/review.json']:
        inputs[p.resolve().as_posix()]=sha(p)
    for p in SOURCE.iterdir():
        if p.is_file():inputs[p.as_posix()]=sha(p)
    synthetic_graphs={p.as_posix():sha(p) for p in (BASE/'synthetic_only_temp').rglob('*.onnx')}
    assert len(synthetic_graphs)==1
    for p,digest in synthetic_graphs.items():inputs[p]=digest
    report=dict(source_preparation_passed=True,preparation_only=True,source_directory=SOURCE.as_posix(),source_sha256=source_map,
        source_derivation_sha256=sha(BASE/'derivation.json'),original_promoted_class_AST_exact=True,
        original_exporter_AST_exact_except_graph_name=True,source_changes=derivation['exact_replacements'],
        synthetic_tests=count,tests_failed=0,tests_skipped=0,runtime=runtime,
        graph_nodes=20,graph_initializers=10,graph_opset=17,graph_ir_version=8,graph_input=['float32',None,1323],graph_output=['float32',None,23],
        original_parameter_dtype='float32',internal_dtype='float64',future_ordinary_step=81000,
        unchanged_parity_tolerance_rad=1e-5,final_admission_backends=['CPU64','GPU64','ORT64'],final_GPU32_drift_is_diagnostic=True,
        initialization_GPU32_comparison_gate_preserved=True,synthetic_graphs=synthetic_graphs,input_sha256=inputs,
        actual_checkpoint_loads=0,task_arrays_loaded=0,Torch_forward_calls=0,ORT_sessions=0,ORT_calls=0,
        cuda_initialized=False,native_steps=0,optimizer_updates=0,actual_fit_created=False,actual_export_created=False,
        actual_training_selected=False,execution_clearance=False,
        limitations=['Parameter/graph evidence only; actual forward numerical parity is not tested or qualified.',
            'Synthetic ONNX artifact is not a trained head and cannot serve as release, witness or canonical input.',
            'Future immutable checkpoint/data/fit release subjects, numerical diagnostics and native acceptance remain required.'])
    write(BASE/'source_preparation.json',report)
    print(json.dumps(dict(source_preparation_sha256=sha(BASE/'source_preparation.json'),source_sha256=source_map,tests=count)))

if __name__=='__main__':main()
