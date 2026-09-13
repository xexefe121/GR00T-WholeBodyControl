from pathlib import Path
ROOT=Path(__file__).resolve().parent; NEW=ROOT.parent
s=(NEW/'direct_target_response_fit_audit_root_review_v3/review_concrete.py').read_text(encoding='utf-8-sig')
replacements={
'direct_target_response_balanced_fit_independent_v3':'direct_target_width512_fit_independent_v1',
'7a09feb170fc2f0423d55ce471a82bc07d00f0e5fcce6853a1d428462879ffc4':'7e0d079af78b680331ec83cb294073fe8ae2f3382b4fa2b3e6a2b89d10ca397a',
'0f5b558c6e08bf50923fe8ee03892108d9914ff79ddf6e9be99ed82de146e81e':'e53a061fb672a1bb6e1c518cf284115001abac187d4e17226dccdb37e71bc864',
'saved_response_balanced_warm_only':'saved_width512_warm_only',
'direct_target_causal_response_balanced_student_v2':'direct_target_causal_width512_student_v1',
'a6b75e9df7864e46a5ed0ca84efe05fc0fa2e020b25766d6077b3e6bc744b8d3':'18c66ebdb2af7138858c08d012c5d3cc7819c1bb2bb5a5b523e488c35e4ed6a2',
"len(launch['input_sha256'])==38":"len(launch['input_sha256'])==41",
"len(request['source_sha256'])==13":"len(request['source_sha256'])==16",
'launch_pin_count=38':'launch_pin_count=41',"'pins':38":"'pins':41"}
for a,b in replacements.items(): assert a in s,a; s=s.replace(a,b)
with (ROOT/'review_concrete.py').open('x',encoding='utf-8') as f:f.write(s)
