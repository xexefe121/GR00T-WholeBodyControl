from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
B=Path(__file__).resolve().parent;BASE=B.parent/'independent_native_stepper_v1'
source=BASE/'source_draft_v3'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
derivation=json.loads((B/'v3_source_derivation.json').read_text())
for name,digest in derivation['old_source_pins'].items():assert sha(BASE/'source_draft_v2'/name)==digest
for name,digest in derivation['new_source_pins'].items():assert sha(source/name)==digest
tree=ET.parse(source/'tests_root_v3.xml')
cases=tree.findall('.//testcase')
assert len(cases)==24 and not tree.findall('.//failure') and not tree.findall('.//error')
files=list(source.glob('*.py'))+[source/'tests_root_v3.xml',BASE/'source_preparation_v2.json',BASE/'V2_DESIGN.md',B/'v3_source_derivation.json',B/'closed_restore_v2.json']
report=dict(source_review_passed=True,native_equivalence_cleared=False,source_directory=str(source),
    source_and_evidence_pins={str(p):sha(p) for p in files},
    v2_findings_resolved=['full expected MJB replaces partial physics whitelist','failure capture retains full291/actual forces fieldwise without another step'],
    additional_root_finding=dict(description='closed uninitialized adapter permitted set/forward/set after exit verification',
        evidence_sha256=sha(B/'closed_restore_v2.json'),corrected_in='source_draft_v3/native_stepper.py',
        change='reject closed or faulted adapter before restore_attempted or any native state write'),
    tests_passed=24,real_model_constructions=0,real_native_serializations=0,real_native_steps=0,model_inference_calls=0,
    limitations=['source and fake API verification only; actual full MJB serialization and oracle step equality still need selected native experiment',
        'MJB excludes transient reverted mutations, direct native callback/collision table writes and loaded implementation identity',
        'MJB is platform/version/padding specific and stricter for visual/stat changes',
        'foundation exact repeated clock comparison is stricter than original oracle 1e-10 tolerance',
        'no independent real process timing or balancing/rearm qualification'])
out=B/'review_v3.json';assert not out.exists()
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'review':str(out),'sha256':sha(out),'tests':24,'source_review_passed':True,'native_equivalence_cleared':False}))
