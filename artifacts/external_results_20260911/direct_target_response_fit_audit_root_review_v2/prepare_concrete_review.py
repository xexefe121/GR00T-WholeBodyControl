from pathlib import Path
BASE=Path(__file__).resolve().parent
s=(BASE.parent/'direct_target_response_fit_audit_root_review_v1/review_concrete.py').read_text(encoding='utf-8-sig')
for old,new in [
 ('direct_target_response_balanced_fit_independent_v1','direct_target_response_balanced_fit_independent_v2'),
 ('f23b1c471178a6dea64612c60ab37206c26848daa14172d588b39e96668c163d','e1e629307425a43b25d54c3c7ae7039c2c174bfd213e5171e5f71efa59a424de'),
 ('4f007153b743de2bad653cdedf08094990d7f384b7bcd378f3c98edd3ff61dfb','7fce5b4880a201c916b9abfebf223e55245382b956594a6f2c28b738ddf759da'),
 ("len(launch['input_sha256'])==36","len(launch['input_sha256'])==37"),
 ("len(request['source_sha256'])==11","len(request['source_sha256'])==12"),
 ("BASE/'audit_process_v1'","BASE/'process_v1'"),
 ('launch_pin_count=36','launch_pin_count=37'),("'pins':36","'pins':37")]:
    assert old in s,old
    s=s.replace(old,new)
with (BASE/'review_concrete.py').open('x',encoding='utf-8') as f:f.write(s)
