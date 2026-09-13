"""Freeze metadata-only repair preparation after synthetic and JSON schema checks."""
from pathlib import Path
import ast,hashlib,json,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent;NEW=BASE.parent;OLD=NEW/'direct_target_response_balanced_fit_independent_v1'
SOURCE=BASE/'source_prepared_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
class NormalizeLabels(ast.NodeTransformer):
    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('ENERGY_PROVENANCE_RULE','PRODUCER_WEIGHT_RULE'):return None
        return self.generic_visit(node)
    def visit_Name(self,node):
        return ast.Constant('Emean/Eg') if node.id in ('ENERGY_PROVENANCE_RULE','PRODUCER_WEIGHT_RULE') else node
    def visit_ImportFrom(self,node):
        node.names=[n for n in node.names if n.name!='PRODUCER_WEIGHT_RULE'];return node
def main():
    oldprep=read(OLD/'source_preparation.json');changes=[];unchanged={}
    for name,digest in oldprep['source_sha256'].items():
        prior=OLD/'source_prepared_v1'/name;current=SOURCE/name;assert sha(prior)==digest
        if sha(current)==digest:unchanged[name]=digest
        else:
            changes.append(name);assert name in ('audit_balanced_math.py','audit_saved_warm.py')
            old=ast.parse(prior.read_text());new=NormalizeLabels().visit(ast.parse(current.read_text()))
            assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
    assert len(changes)==2 and len(unchanged)==9
    helpers={n:sha(BASE/n) for n in ('prepare_audit_request.py','freeze_launch.py','run_audit_durable.ps1','verify_completion.py')}
    for n,d in helpers.items():assert d==sha(OLD/n)
    tests=ET.parse(BASE/'tests_v1.xml').getroot()
    assert len(tests.findall('.//testcase'))==46 and not any(tests.findall('.//'+k) for k in ('failure','error','skipped'))
    schema=read(BASE/'saved_metadata_inspection.json');assert schema['metadata_inspection_passed'] is True and schema['checks']==75
    for p,d in schema['input_sha256'].items():assert sha(p)==d
    assert not (BASE/'audit_request.json').exists() and not (BASE/'results_v1').exists() and not (BASE/'process_v1').exists()
    references=[OLD/'source_preparation.json',OLD/'results_v1/report.json',OLD/'owner_completion.json',
        NEW/'direct_target_response_fit_audit_rule_diagnosis_v1/report.json',
        NEW/'direct_target_response_fit_audit_root_review_v1/combined_review.json',
        NEW/'direct_target_causal_response_balanced_student_v2/source_snapshot_v1/balance_contract.py']
    result=dict(source_preparation_passed=True,preparation_only=True,actual_audit_executed=False,
        source_directory=SOURCE.as_posix(),experiment=(NEW/'direct_target_causal_response_balanced_student_v2').as_posix(),
        source_sha256={p.name:sha(p) for p in sorted(SOURCE.glob('*.py'))},helper_sha256=helpers,
        changed_production_sources=changes,unchanged_source_sha256=unchanged,new_test='test_rule_metadata.py',
        source_AST_equal_after_exact_metadata_label_normalization=True,all_numerical_arithmetic_unchanged=True,
        source_review_scope='Only independent provenance/producer string constants and three producer metadata expectations; energy.rule remains Emean/Eg.',
        reference_sha256={p.as_posix():sha(p) for p in references},
        evidence_sha256={n:sha(BASE/n) for n in ('tests_v1.xml','source_delta.patch','derivation.json','prepare_source.py','record_preparation.py','inspect_saved_metadata.py','inspect_saved_metadata_v2.py','metadata_inspection_preparation_failure.json','saved_metadata_inspection.json')},
        synthetic_tests_passed=46,saved_JSON_metadata_checks_passed=75,task_array_loads=0,task_checkpoint_loads=0,
        task_model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,
        original_failed_audit_preserved=True,actual_numerical_audit_repeated=False,automatic_retry=False)
    with (BASE/'source_preparation.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(preparation_sha256=sha(BASE/'source_preparation.json'),source_count=len(result['source_sha256']),helper_count=len(helpers))))
if __name__=='__main__':main()
