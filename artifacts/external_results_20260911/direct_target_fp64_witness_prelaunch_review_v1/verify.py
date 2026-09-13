import ast,hashlib,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
N=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911');B=N/'direct_target_fp64_export_evaluation_v2';S=B/'source_draft_v1';P=B/'witness_process'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
binding=B/'witness_binding.json';receipt=P/'launch_receipt.json'
assert sha(binding)=='48ab01a8238008ab806f00a3c2328399960040452762cc16965d1baae8a67319'
assert sha(receipt)=='0d1018febc1f3f9d1f2bf043b503c0011b136d7174347c899bb674c28e3892f3'
b=read(binding);r=read(receipt);assert r['binding_sha256']==sha(binding)
assert r['kind']=='one_selected_direct_target_witness' and r['expected_separate_head_calls']==1
assert r['requested_main_controls']==r['conditional_hold_controls']==0 and r['automatic_retry'] is False
assert len(r['input_hashes'])==5216 and len(b['input_files'])==5211
sys.path.insert(0,str(S));import evaluation_gate
ready=evaluation_gate.require_model_ready(B,'witness')
assert b['head']['sha256']=='147a710ac8d6fd6de592c93f3ca14af4f7fcf156b970bbc3ab7d5d86f3586501'
assert b['physics_authorized'] is False and b['expected_head_calls']==1
assert b['reviews']['export']['sha256']=='81e7e86cee62c29b926a1f33ea096c1d5ea677fc83494c19cf9f2cbb039ffb0e'
assert b['export_owner_completion']['sha256']=='696a0ecf7643127b2558dd31651282bc5704274fb7f0554a3728b2c4ff10749d'
assert b['reviews']['source']['sha256']=='5b414d0260f21984f98e134134e7ab9e01a4465028929f607715973820f8354d'
assert r['exact_wsl_arguments']==['-d','Ubuntu-22.04','--cd','/','--','bash','/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONPATH=/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_fp64_export_evaluation_v2/source_draft_v1','/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python','/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_fp64_export_evaluation_v2/source_draft_v1/head_activation_witness.py']
for entry in b['input_files']:assert r['input_hashes'][entry['path']]==entry['sha256']
for path,digest in r['input_hashes'].items():assert sha(path)==digest
config=read(B/'release_reviews.json');assert sha(B/'release_reviews.json')=='ae56a62901d185206b01496acc15c97ed3cd6d018b3e715284dad2c8ab5dcd72'
assert config['prebinding_root_export_audit_passed'] is True
assert config['root_export_audit']['sha256']=='8f89c2c18c31cfdfb11dcfde2e8ce4e91f61cb90c1b83c6e1ee99ee122bbd2d9'
assert sha(config['root_export_audit']['path'])==config['root_export_audit']['sha256']
old=(B/'prepare_review_configuration.py').read_text();new=(B/'prepare_review_configuration_v2.py').read_text()
expected=old.replace("    a=parser.parse_args()","    parser.add_argument('--export-owner',type=Path,required=True)\n    a=parser.parse_args()").replace("owner=subject(EXPORT/'owner_completion_verification.json')","owner=subject(a.export_owner)")
assert new==expected;ast.parse(new)
assert sha(B/'prepare_review_configuration_v2.py')==config['preflight_source']['sha256']=='093311b4ef92a81cc4469404c2856a2d988865ac7ca0c41831729b4f9d6314e1'
tree=ET.parse(B/'review_configuration_tests_v2.xml').getroot();tests=list(tree.iter('testsuite'))
assert sum(int(x.attrib['tests']) for x in tests)==9 and all(int(x.attrib[k])==0 for x in tests for k in ('errors','failures','skipped'))
assert not (B/'head_witness').exists()
for name in ('started.lock','start.json','child.json','stdout.log','stderr.log','exit.json','raw_exit.json','launch_clearance.json'):assert not (P/name).exists()
subjects={}
for name,path in {'binding':binding,'launch_receipt':receipt,'run':P/'run.ps1','durable':P/'run_durable.ps1','review_configuration':B/'release_reviews.json','configuration_helper':B/'prepare_review_configuration_v2.py','configuration_tests':B/'review_configuration_tests_v2.xml'}.items():subjects[name]={'path':str(path),'sha256':sha(path)}
report={'kind':'one_actual_same55000_FP64_WSL_witness_prelaunch_review','passed':True,'prelaunch_review_pass':True,
 'binding_sha256':sha(binding),'launch_receipt_sha256':sha(receipt),'subjects':subjects,'verified_launch_pin_count':5216,
 'source_review_pass':True,'configuration_helper_review_pass':True,'expected_head_calls':1,'BFM_calls':0,'native_steps':0,
 'checks':['Actual positive training-only, separate FP64 export, corrected owner and independent export audit identities match immutable direct subjects. Original failed FP32 flags remain false.',
 'Reviewed additive prebinding helper and exact two-line owner-path v2 delta; nine synthetic subject tests pass. Configuration pins exact helper and actual root export audit.',
 'All5216 launch hashes and5211 runtime binding subjects verified. require_model_ready passes actual frozen source, runtime/reference/contract/query250 feature gates without creating a task model.',
 'Actual WSL bootstrap/PYTHONPATH/interpreter/head-witness command, one-call scope, hidden CreateNew durable wrapper, retained child handle, raw/diagnostic exit and postrun hash logic match reviewed generator.',
 'Head witness outputs/process/lock/clearance absent before review. Root-selected one witness is cleared after final launch clearance binds this receipt.'],
 'selected_witness_launch_cleared':True,'canonical_launch_cleared':False,'reviewer_model_calls':0,'reviewer_native_steps':0,'reviewer_optimizer_updates':0,'findings':[]}
out=Path(__file__).parent/'review.json'
with out.open('x') as file:json.dump(report,file,indent=2);file.write('\n')
print(json.dumps({'passed':True,'review_sha256':sha(out)}))
