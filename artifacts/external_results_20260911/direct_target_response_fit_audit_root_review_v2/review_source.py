"""Root narrow metadata-label repair review; hashes, source and existing tests only."""
from pathlib import Path
import ast,hashlib,json,xml.etree.ElementTree as ET
OUT=Path(__file__).resolve().parent;NEW=OUT.parent
BASE=NEW/'direct_target_response_balanced_fit_independent_v2';OLD=NEW/'direct_target_response_balanced_fit_independent_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(BASE/'source_preparation.json')=='8a34cff1d4a26a21d2da91d044825a4656bd9819cbbc785f3af8ef1edd952e75'
p=read(BASE/'source_preparation.json');oldreview=read(NEW/'direct_target_response_fit_audit_root_review_v1/combined_review.json')
for name,h in p['source_sha256'].items():assert sha(BASE/'source_prepared_v1'/name)==h
for name,h in p['unchanged_source_sha256'].items():assert sha(OLD/'source_prepared_v1'/name)==h
assert len(p['source_sha256'])==12 and len(p['unchanged_source_sha256'])==9
for name,h in p['helper_sha256'].items():assert sha(BASE/name)==sha(OLD/name)==oldreview['helper_sha256'][name]==h
for path,h in p['reference_sha256'].items():assert sha(path)==h
for name,h in p['evidence_sha256'].items():assert sha(BASE/name)==h
class LabelNormalization(ast.NodeTransformer):
    def visit_Assign(self,node):
        if any(isinstance(x,ast.Name) and x.id in ('ENERGY_PROVENANCE_RULE','PRODUCER_WEIGHT_RULE') for x in node.targets):return None
        return self.generic_visit(node)
    def visit_ImportFrom(self,node):
        node.names=[n for n in node.names if n.name!='PRODUCER_WEIGHT_RULE'];return node
    def visit_Name(self,node):
        if node.id in ('ENERGY_PROVENANCE_RULE','PRODUCER_WEIGHT_RULE'):return ast.Constant('Emean/Eg')
        return node
for name in p['changed_production_sources']:
    original=ast.parse((OLD/'source_prepared_v1'/name).read_text(encoding='utf-8-sig'))
    repaired=ast.parse((BASE/'source_prepared_v1'/name).read_text(encoding='utf-8-sig'))
    assert ast.dump(original,include_attributes=False)==ast.dump(LabelNormalization().visit(repaired),include_attributes=False),name
xml=ET.parse(BASE/'root_tests.xml').getroot();suites=list(xml) if xml.tag!='testsuite' else [xml]
assert sum(int(s.attrib['tests']) for s in suites)==46
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
report=dict(source_review_pass=True,helper_review_pass=True,reviewer='root',source_preparation_sha256=sha(BASE/'source_preparation.json'),source_sha256=p['source_sha256'],helper_sha256=p['helper_sha256'],prior_source_review_sha256=sha(NEW/'direct_target_response_fit_audit_root_review_v1/combined_review.json'),root_synthetic_tests=46,root_tests_sha256=sha(BASE/'root_tests.xml'),unchanged_source_count=9,unchanged_helper_count=4,numerical_AST_unchanged_after_only_literal_label_normalization=True,reviewed_semantics=['Independent energy provenance label remains Emean/Eg; checkpoint, report and reconstructed metrics require exact pinned producer label.','Seven actual-expression/metric regressions reject abbreviated producer labels and long energy provenance label.','No model parameters, saved predictions, data, numerical arithmetic, parity tolerance or fit inputs changed.','Failed original audit remains preserved; new actual request and launch require separate concrete binding check.'],actual_audit_selected=False,task_array_loads=0,task_checkpoint_loads=0,model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,writer_sha256=sha(__file__))
with (OUT/'review.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'source_review_pass':True,'sha256':sha(OUT/'review.json')}))
