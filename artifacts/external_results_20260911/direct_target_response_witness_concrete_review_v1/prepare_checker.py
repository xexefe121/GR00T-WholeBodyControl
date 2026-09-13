"""Adapt qualified concrete witness review only; never executes a packet check."""
from pathlib import Path
import ast,difflib,hashlib,json
HERE=Path(__file__).resolve().parent;NEW=HERE.parent
OLD=NEW/'direct_target_context_witness_concrete_review_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(OLD/'review_concrete.py')==json.loads((OLD/'review.json').read_text())['writer_sha256']
old=(OLD/'review_concrete.py').read_text();s=old;changes=[]
def replace(a,b):
    global s
    assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b);changes.append(a[:100])
s=s.replace('causal68000','causal71000').replace(' == 68000',' == 71000').replace('ordinary_final_step=68000','ordinary_final_step=71000')
s=s.replace('direct_target_causal_context_evaluation_v2','direct_target_causal_response_evaluation_v1')
replace('import hashlib, json, re','import hashlib, json, re, argparse')
replace("BINDING_SHA = '1502f0365a29c5ec8b21b2547f158a37d38999fb90260b34371e89a319e4006d'\nLAUNCH_SHA = 'f3772bc07cf1a46b4f212cafd8f13ec4ad9f1c81388b35b95e3753027982b1d8'", "parser=argparse.ArgumentParser(description=__doc__)\nfor name in ('binding-sha256','launch-sha256'):parser.add_argument('--'+name,required=True)\nfor name in ('binding-pins','launch-pins'):parser.add_argument('--'+name,type=int,required=True)\nargs=parser.parse_args()\nBINDING_SHA,LAUNCH_SHA=args.binding_sha256,args.launch_sha256\nassert args.binding_pins>0 and args.launch_pins==args.binding_pins+5")
for a,b in [('c8002f3667de0627695262338bb5bd7a0ffeadfe85c9f003b6baa5b8d53e5dc4','0ae9699fcb2b70ff33948223e9630a0f8692751baf2fdb27bbb191e7d76d5480'),
 ('949b5124b0f452a917c87660a79f294e5276a8b20bc3da4c2a516d6d88dd3a29','e40aca20c419219a8e6d1e4630ef5e4442ac7c1f492d85b009bc94b1460a0121'),
 ('ab65dd600ac6ad8e1ff7093cc8e0a56f40285d09eca55d212124f08bd789c2af','427a91b32475e7f59212954d109865227f6b523c3e43ea9037c14462300a50ff'),
 ('8e6f1bfde3249bf8b8ada99a1c395c6cc3df57ce05f59ab0d91192aef466c979','a6b75e9df7864e46a5ed0ca84efe05fc0fa2e020b25766d6077b3e6bc744b8d3'),
 ('d61915c1bf30b16431660134be6057855bbc7befeb620630a5b36dd506738f5c','c70e90adab31efd2abcebd9817bf3c3c1ecc84cde546e080f6367c710633423a'),
 ('10d238198e2d7826c35eaa9bd4a13ff2e05050a7f601c1f212b585c4a90eaefd','395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d'),
 ('7f1806a2132ea3879457a3bb2705105b3075835417151cc9003143974bc99dd9','0899cdf7b9d56481ab71ea4d9c61f86d1104c4d5238d8df4d7e1c5e4e5f7d7a3')]:replace(a,b)
replace("SOURCE_SHA = 'e40aca20c419219a8e6d1e4630ef5e4442ac7c1f492d85b009bc94b1460a0121'", "SOURCE_SHA = 'e40aca20c419219a8e6d1e4630ef5e4442ac7c1f492d85b009bc94b1460a0121'\nHELPER_SHA='b95fa1160498af88cf3fe38c80aaf12b8707ac3d18f25ed69fdfe54c9920d16f'")
replace("'paired_completion_passed'","'completion_passed'")
replace("roles = audit['condition_subjects']['causal']\nassert len(roles) == 18", "roles = release['subjects']\nassert len(roles) == 16\nassert set(roles)==set(audit['direct_subject_sha256'])\nassert release['selected_main_controls']==1569 and release['conditional_hold_controls']==250\nconfig=read(BASE/'release_reviews.json')\nhelper_entry=config['launch_helper_review'];helper=entry(helper_entry)\nassert helper_entry['sha256']==HELPER_SHA and helper['passed'] is True\nassert len(source['source_sha256'])==37")
replace("assert owner['direct_subject_sha256'][role] == release['direct_subject_sha256'][role] == subject['sha256'], role", "assert owner['direct_subject_sha256'][role] == release['direct_subject_sha256'][role] == audit['direct_subject_sha256'][role] == subject['sha256'], role")
replace("assert release['source_and_helper_review_sha256'] == SOURCE_SHA", "assert release['source_review_sha256']==SOURCE_SHA and release['helper_review_sha256']==HELPER_SHA")
replace("direct_target_context_pair_fit_independent_v1/owner_completion.json","direct_target_response_balanced_fit_independent_v3/owner_completion.json")
replace("assert len(b['input_files']) == 5225 and len(pins) == 5230", "assert len(b['input_files']) == args.binding_pins and len(pins) == args.launch_pins\nassert pins[(BASE/'release_reviews.json').as_posix()]==sha(BASE/'release_reviews.json')\nassert pins[helper_entry['path']]==HELPER_SHA")
replace("for name, digest in source['helper_sha256'].items():", "for name, digest in helper['helper_sha256'].items():")
replace('checked_binding_pins=5225, checked_input_pins=5230','checked_binding_pins=args.binding_pins, checked_input_pins=args.launch_pins')
replace('release_subjects_checked=18','release_subjects_checked=16')
replace("helper_files_checked=len(source['helper_sha256']), source_review_sha256=SOURCE_SHA", "helper_files_checked=len(helper['helper_sha256']), helper_review_sha256=HELPER_SHA, source_review_sha256=SOURCE_SHA")
ast.parse(s)
with (HERE/'review_concrete.py').open('x') as f:f.write(s)
with (HERE/'source_delta.patch').open('x') as f:f.write(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='qualified68000/review_concrete.py',tofile='review_concrete.py')))
with (HERE/'preparation.json').open('x') as f:json.dump(dict(source_preparation_passed=True,actual_checker_executed=False,
    source_sha256=sha(HERE/'review_concrete.py'),parent_checker_sha256=sha(OLD/'review_concrete.py'),derivation_sha256=sha(HERE/'source_delta.patch'),
    changes=changes,release_sha256='0ae9699fcb2b70ff33948223e9630a0f8692751baf2fdb27bbb191e7d76d5480',
    parameters_require_actual_binding_and_launch_subjects=True,task_model_calls=0,ORT_calls=0,native_steps=0,dispatch_performed=False),f,indent=2)
print(json.dumps(dict(checker_sha256=sha(HERE/'review_concrete.py'),preparation_sha256=sha(HERE/'preparation.json'))))
