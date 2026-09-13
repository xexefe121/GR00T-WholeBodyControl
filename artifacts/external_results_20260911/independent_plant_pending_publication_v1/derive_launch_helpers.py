"""Add new helper namespace only; reviewed producer/auditor sources stay immutable."""
from pathlib import Path
import hashlib,json,difflib

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
OLD=NEW/'independent_plant_clock_timeout_correction_v1'
AUDIT=NEW/'independent_plant_pending_publication_saved_actual_v1'
OLD_AUDIT=NEW/'independent_plant_clock_saved_actual_v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(src,dst):
    with dst.open('xb') as f:f.write(src.read_bytes())
def changed(src,dst,replacements):
    text=src.read_text()
    for a,b in replacements:
        if a not in text:raise ValueError('Missing derivation input '+a)
        text=text.replace(a,b)
    with dst.open('x',encoding='utf-8') as f:f.write(text)
    return ''.join(difflib.unified_diff(src.read_text().splitlines(True),text.splitlines(True),fromfile=str(src),tofile=str(dst)))

def main():
    AUDIT.mkdir(exist_ok=False)
    copies={}
    for name in ('prepare_stage_preserved_template.py','stage_verdict.py','verify_completion.py'):
        copy(OLD/name,BASE/name);copies[str(BASE/name)]={'original':str(OLD/name),'sha256':sha(BASE/name)}
    diff=changed(OLD/'prepare_clock_stage.py',BASE/'prepare_clock_stage.py',[
        ("    validate_request(request)\n", "    validate_request(request)\n    from prepare_concrete_packet import validate_retry_contract\n    validate_retry_contract(request)\n")])
    for name in ('verify_completion.py','preserved_saved_audit_template.ps1.txt'):
        copy(OLD_AUDIT/name,AUDIT/name);copies[str(AUDIT/name)]={'original':str(OLD_AUDIT/name),'sha256':sha(AUDIT/name)}
    diff+=changed(OLD_AUDIT/'prepare_launch.py',AUDIT/'prepare_launch.py',[
        ('Frozen source-audit-v3','Frozen retry-aware source-draft-v2'),
        ('independent_plant_clock_saved_root_review_v1/source_audit_v3','independent_plant_pending_publication_saved_audit_v1/source_draft_v2'),
        ('independent_plant_clock_saved_root_review_v1/root_source_review_v3.json','independent_pending_publication_saved_root_review_v1/review.json'),
        ('278c509418ef1c7a1be655693a840217dd8c1894a6a6c33f0757a712ad8b9ac7','81c4090dd6035fc4b39064fd98e7412af87e2d0355a4a3abd260d19ffb4faae0')])
    diff+=changed(OLD_AUDIT/'test_launch.py',AUDIT/'test_launch.py',[
        ('fixed_v3_source','fixed_retry_v2_source'),
        ('source_audit_v3','source_draft_v2'),
        ('independent_plant_clock_saved_actual_v1','independent_plant_pending_publication_saved_actual_v1'),
        ('independent_plant_clock_saved_root_review_v1','independent_plant_pending_publication_saved_audit_v1')])
    with (BASE/'helper_derivation.diff').open('x') as f:f.write(diff)
    with (BASE/'helper_derivation.json').open('x') as f:json.dump({'unchanged_copies':copies,'source_only':True,'actual_dispatch':False},f,indent=2)

if __name__=='__main__':main()
