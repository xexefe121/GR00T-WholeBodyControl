from pathlib import Path
BASE=Path(__file__).resolve().parent
s=(BASE.parent/'direct_target_response_fit_audit_root_review_v2/review_concrete.py').read_text(encoding='utf-8-sig')
for old,new in [
 ('direct_target_response_balanced_fit_independent_v2','direct_target_response_balanced_fit_independent_v3'),
 ('e1e629307425a43b25d54c3c7ae7039c2c174bfd213e5171e5f71efa59a424de','7a09feb170fc2f0423d55ce471a82bc07d00f0e5fcce6853a1d428462879ffc4'),
 ('7fce5b4880a201c916b9abfebf223e55245382b956594a6f2c28b738ddf759da','0f5b558c6e08bf50923fe8ee03892108d9914ff79ddf6e9be99ed82de146e81e'),
 ("len(launch['input_sha256'])==37","len(launch['input_sha256'])==38"),
 ("len(request['source_sha256'])==12","len(request['source_sha256'])==13"),
 ('launch_pin_count=37','launch_pin_count=38'),("'pins':37","'pins':38")]:
    assert old in s,old
    s=s.replace(old,new)
with (BASE/'review_concrete.py').open('x',encoding='utf-8') as f:f.write(s)
