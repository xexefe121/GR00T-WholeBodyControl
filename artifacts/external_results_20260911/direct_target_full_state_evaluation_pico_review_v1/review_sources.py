"""Independent source/fixture review only. No task arrays, models or physics."""
import ast,copy,difflib,hashlib,json,sys
from pathlib import Path
import xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
TARGET=NEW/'direct_target_full_state_evaluation_v1';SOURCE=TARGET/'source_draft_v1'
ORIGINAL=NEW/'direct_target_fp64_export_evaluation_v2/source_draft_v1'
sys.path.insert(0,str(SOURCE))
from test_release_gate import fixture
from export_release_gate import validate_release


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def main():
    preparation=TARGET/'source_preparation.json'
    assert sha(preparation)=='13c24f8987a1cd9e1c62bc33e7b9cafb62a8c728db4c4fa28f3b58c9ee716326'
    prep=read(preparation);subjects={name:sha(SOURCE/name) for name in prep['source_sha256']}
    assert subjects==prep['source_sha256'] and len(subjects)==32
    assert len(prep['unchanged_files'])==29
    original_subjects={name:sha(ORIGINAL/name) for name in prep['unchanged_files']}
    assert all(subjects[name]==digest for name,digest in original_subjects.items())
    old=(ORIGINAL/'evaluation_gate.py').read_text();new=(SOURCE/'evaluation_gate.py').read_text()
    before=ast.parse(old);after=ast.parse(new)
    nodes_old={n.name:ast.dump(n,include_attributes=False) for n in before.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    nodes_new={n.name:ast.dump(n,include_attributes=False) for n in after.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert set(nodes_old)==set(nodes_new)
    changed=[name for name in nodes_old if nodes_old[name]!=nodes_new[name]]
    assert changed==['require_model_ready']
    diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='reviewed55000/evaluation_gate.py',tofile='65000/evaluation_gate.py'))
    with (BASE/'evaluation_gate.diff').open('x',encoding='utf-8') as f:f.write(diff)
    # Additional fake-chain probes do not load actual release subjects or arrays.
    probes=[]
    for bad in [float('nan'),float('inf'),-1e-12]:
        binding,paths,source,records,context=fixture();records[paths['fit_report']]['max_preclip_error_rad']=bad
        try:validate_release(binding,paths,source,'witness',**context)
        except AssertionError:probes.append('nonfinite/negative parity rejected')
        else:raise AssertionError('invalid parity accepted')
    for name in ('full_state_generation_request','full_state_data_audit','checkpoint','head','normalization'):
        binding,paths,source,records,context=fixture()
        records[binding['root_training_audit']['path']]['input_sha256']['/mnt/e/synthetic/'+name+'.json']='0'*64
        try:validate_release(binding,paths,source,'evaluation',**context)
        except AssertionError:probes.append('wrong consumed '+name+' rejected')
        else:raise AssertionError('wrong consumed subject accepted')
    suites=ET.parse(BASE/'synthetic_tests.xml').getroot()
    assert len(suites.findall('.//testcase'))==55 and not any(suites.findall('.//'+name) for name in ('failure','error','skipped'))
    assert not any(name.split('.')[0] in ('mujoco','onnxruntime','torch','mjbatch') for name in sys.modules)
    assert {name:sha(SOURCE/name) for name in subjects}==subjects and sha(preparation)==prep_sha
    evidence={str(preparation):sha(preparation),str(TARGET/'DESIGN.md'):sha(TARGET/'DESIGN.md'),
        str(BASE/'synthetic_tests.xml'):sha(BASE/'synthetic_tests.xml'),str(BASE/'evaluation_gate.diff'):sha(BASE/'evaluation_gate.diff'),
        str(Path(__file__)):sha(__file__)}
    result={'passed':True,'source_review_pass':True,'preparation_only':True,'findings':[],
        'source_directory':SOURCE.as_posix(),'source_sha256':subjects,'unchanged_original_sha256':original_subjects,
        'source_preparation_sha256':sha(preparation),'synthetic_tests_passed':55,'additional_fake_chain_probes':len(probes),
        'actual_task_array_reads':0,'actual_head_calls':0,'native_steps':0,'optimizer_updates':0,
        'evidence_sha256':evidence,'reviewed_contracts':[
        'Only require_model_ready AST changed in evaluation_gate; unchanged witness parity gate and29 runtime/native copies.',
        'Pure1000 features, float32 span promoted before float64 output reconstruction, appliedinverse history and original phases unchanged.',
        'One new actual WSL batch1 witness required; original1569 and conditionalcontinuous250 remain mandatory.',
        'Actual complete65000/fresh10000-step fit, same-weight FP64 export and original1e-5 gate required.',
        '354612 full58 endpoints,140622 oldoverlap,213990 independentnew and9904+3054 other populations bound.',
        'Thirteen literal direct subjects bind owner/release; independentfit input membership and actual data owner/report are explicit.',
        'Source review hash gate binds actual source modules; no placeholder or original55000 failed-release bypass.'],
        'limitations':['Source-only review; no future actual fit/export/owner or concrete launcher cleared.',
                       'Behavioral and timing qualification require separately selected actual trials and independent audits.']}
    write(BASE/'review.json',result);print(json.dumps({'review_sha256':sha(BASE/'review.json'),'passed':True,'sources':32,'unchanged':29,'tests':55,'probes':8}))


prep_sha='13c24f8987a1cd9e1c62bc33e7b9cafb62a8c728db4c4fa28f3b58c9ee716326'
if __name__=='__main__':main()
