from pathlib import Path
import json
base=Path(__file__).resolve().parent
s=(base/'review_concrete.py').read_text()
old="assert actual==render(sha(rp))==(BASE/'template_preview.ps1.txt').read_text()"
new="assert actual==render(sha(rp))\nassert (BASE/'template_preview.ps1.txt').read_text()==render('a'*64)\nassert actual==(BASE/'template_preview.ps1.txt').read_text().replace('a'*64,sha(rp))"
assert s.count(old)==1
with (base/'review_concrete_v2.py').open('x',encoding='utf-8') as f:f.write(s.replace(old,new))
with (base/'review_v1_metadata_failure.json').open('x',encoding='utf-8') as f:json.dump(dict(kind='root_metadata_assertion',reason='Preview intentionally contains a*64 request placeholder, actual contains concrete request hash. Original assertion incorrectly required byte equality before substitution.',production_changed=False,actual_audit_started=False,helper_tests_started=False,model_calls=0,native_steps=0),f,indent=2)
