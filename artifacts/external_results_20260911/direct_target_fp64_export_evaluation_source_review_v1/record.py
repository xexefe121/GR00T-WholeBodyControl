import ast,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
N=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911');B=N/'direct_target_fp64_export_evaluation_v2';V=N/'direct_target_fp64_export_evaluation_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(B/'source_preparation.json')=='5866dc9107a289565c749fa88494a588306ce08e93b3ebe503b336fc934d71fc'
p=read(B/'source_preparation.json');v=read(V/'source_preparation.json');d=read(V/'source_derivation.json');pins={}
assert sha(V/'source_preparation.json')=='c4ed92f27a3db9bbb63fbcd5d0569d7b5b1b0ec7074b75cf1232d59ccd2ac89c'
for base,prep in ((V,v),(B,p)):
 for name,digest in prep['source_sha256'].items():
  q=base/'source_draft_v1'/name;assert sha(q)==digest;pins[str(q)]=digest;ast.parse(q.read_text())
 for path,digest in prep['evidence_sha256'].items():assert sha(path)==digest;pins[path]=digest
for name,digest in p['unchanged_helper_sha256'].items():assert sha(B/name)==sha(V/name)==digest
for name in p['source_sha256']:
 if name!='export_release_gate.py':assert sha(B/'source_draft_v1'/name)==sha(V/'source_draft_v1'/name)
for name,entry in d['sources'].items():
 if name in ('export_release_gate.py','evaluation_gate.py'):continue
 assert sha(entry['original'])==entry['original_sha256']==sha(V/'source_draft_v1'/name)
old=Path(d['sources']['evaluation_gate.py']['original']).read_text();derived=old
for change in d['exact_changes']:
 assert derived.count(change['before'])==1;derived=derived.replace(change['before'],change['after'])
assert ast.dump(ast.parse(derived))==ast.dump(ast.parse((V/'source_draft_v1/evaluation_gate.py').read_text()))
old=(V/'source_draft_v1/export_release_gate.py').read_text();new=(B/'source_draft_v1/export_release_gate.py').read_text()
start=new.index("    root_entry=binding['root_training_audit']");end=new.index("    review('dataset'",start)
assert old==new[:start]+"    review('root_training_audit',original)\n"+new[end:]
count=0
for name,test in p['tests'].items():
 assert sha(B/name)==test['sha256'];tree=ET.parse(B/name).getroot();suites=[tree] if tree.tag=='testsuite' else list(tree.iter('testsuite'))
 assert sum(int(s.attrib.get('tests',0)) for s in suites)==test['tests']
 assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'));count+=test['tests']
assert count==84
inventory=read(B/'runtime_inventory.json');assert inventory['recursive_training_hashes'] is False and len(inventory['files'])==5185
for item in inventory['files']:assert sha(item['path'])==item['sha256']
for name in ('source_preparation.json','runtime_inventory.json','source_derivation.json'):pins[str(B/name)]=sha(B/name)
r={'kind':'same55000_fp64_export_evaluation_source_only_review','passed':True,'source_review_pass':True,'preparation_only':True,
 'actual_export_bound':False,'witness_launch_cleared':False,'canonical_launch_cleared':False,
 'source_sha256':{str(B/'source_draft_v1'/k):h for k,h in p['source_sha256'].items()},'input_sha256':pins,
 'subjects':{'driver':{'path':str(B/'source_draft_v1/evaluate_direct_target_student.py'),'sha256':p['source_sha256']['evaluate_direct_target_student.py']},
 'witness':{'path':str(B/'source_draft_v1/head_activation_witness.py'),'sha256':p['source_sha256']['head_activation_witness.py']},
 'release_gate':{'path':str(B/'source_draft_v1/export_release_gate.py'),'sha256':p['source_sha256']['export_release_gate.py']}},
 'checks':['29 prior controller/witness/helper modules byte-exact; evaluation_gate differs only declared bound subjects and release-validator call.',
 'Preserved v1; v2 changes only root evidence schema validation with explicit six path+hash memberships, positive evidence and negative original export/canonical flags.',
 'Training review binds actual root audit plus six originals directly. New export/owner reviews bind actual head/checkpoint/norm/report/request/manifest; original failed flags retained.',
 'Native limits, BFM250 startup, direct1000-feature arithmetic, applied learned prior and terminal handoff remain unchanged. Original1569 plus conditional250 and separate one-call witness remain gated.',
 '84 owner stub checks inspected, 5185 runtime inventory hashes verified. No recursive training-map traversal or new task calls. Actual numerical/export completion and concrete launch receipts remain required.'],
 'model_calls':0,'native_steps':0,'optimizer_updates':0,'findings':[]}
out=Path(__file__).parent/'review.json'
with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(sha(out))
