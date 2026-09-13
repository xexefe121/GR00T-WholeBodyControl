"""Prepare a parameter-bound width81000 witness checker; execute no task."""
from pathlib import Path
import ast, difflib, hashlib, json
HERE=Path(__file__).resolve().parent; NEW=HERE.parent
PRIOR=NEW/'direct_target_response_witness_concrete_review_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=(PRIOR/'review_concrete.py').read_text(encoding='utf-8-sig'); s=old
assert sha(PRIOR/'review_concrete.py')==json.loads((PRIOR/'review.json').read_text())['writer_sha256']
mapping={
'causal71000':'width81000','direct_target_causal_response_evaluation_v1':'direct_target_causal_width512_evaluation_v1',
"('binding-sha256','launch-sha256')":"('binding-sha256','launch-sha256','release-sha256','audit-sha256','audit-owner-sha256')",
"RELEASE_SHA = '0ae9699fcb2b70ff33948223e9630a0f8692751baf2fdb27bbb191e7d76d5480'":"RELEASE_SHA = args.release_sha256",
"AUDIT_SHA = '427a91b32475e7f59212954d109865227f6b523c3e43ea9037c14462300a50ff'":"AUDIT_SHA = args.audit_sha256",
"== '0899cdf7b9d56481ab71ea4d9c61f86d1104c4d5238d8df4d7e1c5e4e5f7d7a3'":"== args.audit_owner_sha256",
'e40aca20c419219a8e6d1e4630ef5e4442ac7c1f492d85b009bc94b1460a0121':'db2a997d44634cb59d7d4bd92408fcf361323d143c6109024b457e5bd4a458de',
'b95fa1160498af88cf3fe38c80aaf12b8707ac3d18f25ed69fdfe54c9920d16f':'647de5513de061fbf231aad4d1d6e728d1b606e38252f946f11f39cfacc7fe36',
'a6b75e9df7864e46a5ed0ca84efe05fc0fa2e020b25766d6077b3e6bc744b8d3':'18c66ebdb2af7138858c08d012c5d3cc7819c1bb2bb5a5b523e488c35e4ed6a2',
'c70e90adab31efd2abcebd9817bf3c3c1ecc84cde546e080f6367c710633423a':'8a4c394b8836dd763d6eb1c5beac73bf49921bcd59fc5f04850b377d0e843044',
'395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d':'825f86468fcc83913066a9a818f4f438d080e054e17aa4aaadc1451974b8688e',
'direct_target_response_balanced_fit_independent_v3/owner_completion.json':'direct_target_width512_fit_independent_v1/owner_completion.json',
' == 71000':' == 81000','ordinary_final_step=71000':'ordinary_final_step=81000','[1323, 256, 256, 23]':'[1323, 512, 512, 23]',
"len(source['source_sha256'])==37":"len(source['source_sha256'])==38","reviewer='review_continuation'":"reviewer='root'"}
for a,b in mapping.items():assert a in s,a; s=s.replace(a,b)
ast.parse(s)
with (HERE/'review_concrete.py').open('x',encoding='utf-8') as f:f.write(s)
with (HERE/'source_delta.patch').open('x',encoding='utf-8') as f:f.write(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='qualified71000/review_concrete.py',tofile='width81000/review_concrete.py')))
with (HERE/'preparation.json').open('x',encoding='utf-8') as f:json.dump(dict(source_preparation_passed=True,actual_checker_executed=False,source_sha256=sha(HERE/'review_concrete.py'),parent_checker_sha256=sha(PRIOR/'review_concrete.py'),derivation_sha256=sha(HERE/'source_delta.patch'),parameters_require_actual_binding_launch_release_audit_and_owner=True,task_model_calls=0,ORT_calls=0,native_steps=0,dispatch_performed=False),f,indent=2)
print(json.dumps(dict(checker_sha256=sha(HERE/'review_concrete.py'),preparation_sha256=sha(HERE/'preparation.json'))))
