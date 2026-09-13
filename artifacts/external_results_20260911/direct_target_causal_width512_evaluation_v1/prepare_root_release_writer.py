from pathlib import Path
BASE=Path(__file__).resolve().parent; NEW=BASE.parent
s=(NEW/'direct_target_causal_response_evaluation_v1/write_root_final_release.py').read_text(encoding='utf-8-sig')
replacements={
'ordinary71000':'ordinary81000','==71000':'==81000','ordinary_final_step=71000':'ordinary_final_step=81000',
'a6b75e9df7864e46a5ed0ca84efe05fc0fa2e020b25766d6077b3e6bc744b8d3':'18c66ebdb2af7138858c08d012c5d3cc7819c1bb2bb5a5b523e488c35e4ed6a2',
'f918f1dfe675aae01e25307d302ce6c54f599fb7f0f5d1fa3fd943a179b5aded':'b5979023fe9bceb14e0d54cf04bf727fbc413d5a1443b39018e149443b418de6',
'395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d':'825f86468fcc83913066a9a818f4f438d080e054e17aa4aaadc1451974b8688e',
'c70e90adab31efd2abcebd9817bf3c3c1ecc84cde546e080f6367c710633423a':'8a4c394b8836dd763d6eb1c5beac73bf49921bcd59fc5f04850b377d0e843044',
'direct_target_causal_response_evaluation_independent_review_v1':'direct_target_width512_evaluation_independent_review_v2',
'e40aca20c419219a8e6d1e4630ef5e4442ac7c1f492d85b009bc94b1460a0121':'db2a997d44634cb59d7d4bd92408fcf361323d143c6109024b457e5bd4a458de',
'Fixed ordinary81000 endpoint reduces saved nominal, balanced response and physical losses. Response still exceeds zero-response baseline. Select one full original canonical simulation to measure stability, with no numerical or physical gate relaxation.':'Fixed ordinary81000 width512 endpoint reduces saved nominal, balanced response and physical losses after10000 updates. Balanced response still exceeds zero-response baseline. Select one full original canonical simulation to measure stability and actual runtime, with original numerical and physical gates.'}
for a,b in replacements.items(): assert a in s,a; s=s.replace(a,b)
needle="    assert fit['ordinary_final_step']==81000 and fit['optimization_completed'] is fit['numerical_gate_passed'] is True\n"
assert needle in s
s=s.replace(needle,needle+"    assert fit['hidden_width']==512 and fit['architecture']==[1323,512,512,23] and fit['optimizer_step']==16000\n")
with (BASE/'write_root_final_release.py').open('x',encoding='utf-8') as f:f.write(s)
