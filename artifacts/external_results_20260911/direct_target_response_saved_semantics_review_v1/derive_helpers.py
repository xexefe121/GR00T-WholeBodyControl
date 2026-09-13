"""Preserve existing durable implementation; change future metadata references only."""
from derive_sources import BASE,OLD,one,save

def main():
    text=(OLD/'prepare_audit_stage.py').read_text()
    text=one(text,"REVIEW=NEW/'direct_target_context_saved_semantics_root_review_v1/review.json'\nREVIEW_SHA='73517c40584eeda0f7982f0f0653da3aecd949fc28dda85b091d9431799db89d'\n",'')
    text=one(text,"INVENTORY_SHA='4296f2429a3483797c90589213f466e6912801bb1ee05f4f2c158caa69c7b07f'","INVENTORY_SHA='8b11bcdb646779089ab1118468b8753b06317619455e02e8896aaaceb97742a3'")
    text=one(text,"HELPERS=('prepare_audit_stage.py','verify_completion.py','test_audit_stage.py','record_stage_preparation.py')","HELPERS=('prepare_audit_stage.py','verify_completion.py','test_audit_stage.py','derive_helpers.py')")
    text=one(text,"    assert sha(REVIEW)==REVIEW_SHA==request['source_review_sha256']","    REVIEW=local(request['paths']['audit_source_review']);REVIEW_SHA=request['source_review_sha256']\n    assert sha(REVIEW)==REVIEW_SHA")
    save('prepare_audit_stage.py',text)
    path=BASE/'test_audit_stage.py'
    text=one(path.read_text(),"text.count('direct_target_context_saved_semantics_review_v1')","text.count('direct_target_response_saved_semantics_review_v1')")
    path.write_text(text,encoding='utf-8',newline='\n')

if __name__=='__main__':main()
