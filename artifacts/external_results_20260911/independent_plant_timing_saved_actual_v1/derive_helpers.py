"""Source-only copy of completed saved-audit process/owner helpers."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent;NEW=BASE.parent;OLD=NEW/'independent_plant_pending_result_saved_actual_v1'
def put(path,text):
    with path.open('x',encoding='utf-8',newline='') as f:f.write(text)
def replace(text,old,new):
    assert text.count(old)==1,(old,text.count(old));return text.replace(old,new)
def main():
    preserved=BASE/'helper_original_v1';preserved.mkdir(exist_ok=False)
    names=('prepare_launch.py','verify_completion.py','test_launch.py','record_helper_preparation.py',
        'preserved_saved_audit_template.ps1.txt','check_template.ps1')
    for name in names:
        with (preserved/name).open('xb') as f:f.write((OLD/name).read_bytes())
    for name in ('verify_completion.py','preserved_saved_audit_template.ps1.txt','check_template.ps1'):
        with (BASE/name).open('xb') as f:f.write((OLD/name).read_bytes())
    text=(OLD/'prepare_launch.py').read_text()
    text=replace(text,"SOURCE=NEW/'independent_plant_pending_result_saved_audit_v1/source_draft_v2'","SOURCE=NEW/'independent_plant_timing_saved_audit_v1/source_draft_v2'")
    text=replace(text,"SOURCE_REVIEW=NEW/'independent_pending_result_saved_audit_review_v1/review.json'\nSOURCE_REVIEW_SHA='009d21d7dcc8b347da5d0f511ae2d2c24c22774eef84f8ba12f1e7101cb37a48'","SOURCE_PREPARATION=SOURCE.parent/'source_preparation_v2.json'\nSOURCE_PREPARATION_SHA='78d7855373d409fcd6c3a44edef433d6be520faaaf956a739483dad2b8dd6ae8'")
    start=text.index("    q=read(request);assert q['kind']=='independent_saved_clock_audit'")
    end=text.index("    prep=read(BASE/'helper_preparation.json')",start)
    text=text[:start]+"    q=read(request);assert q['kind']=='independent_saved_clock_audit'\n    source_review=Path(q['source_review']['path']).resolve();source_review_sha=sha(source_review)\n    review=read(source_review)\n    assert sha(SOURCE_PREPARATION)==SOURCE_PREPARATION_SHA\n    source_map=read(SOURCE_PREPARATION)['source_sha256']\n    validate_source_review(q,source_review,source_review_sha,review,source_map)\n    for name,digest in source_map.items():assert sha(SOURCE/name)==digest\n"+text[end:]
    text=replace(text,"BASE/'template_parse.json',SOURCE_REVIEW,Path('C:/Windows/System32/wsl.exe')","BASE/'template_parse.json',source_review,SOURCE_PREPARATION,Path('C:/Windows/System32/wsl.exe')")
    text=replace(text,'source_review_sha256=SOURCE_REVIEW_SHA,','source_review_sha256=source_review_sha,')
    helper='''def validate_source_review(request,path,digest,review,source_map):
    role=request['source_review']
    assert Path(role['path']).resolve()==Path(path).resolve() and role['sha256']==digest
    assert review.get('passed') is True and review.get('source_review_pass') is True
    assert review.get('source_preparation_sha256')==SOURCE_PREPARATION_SHA
    assert review.get('source_sha256')==source_map==request['source_sha256'] and len(source_map)==25
    pins={}
    for key,value in request['input_sha256'].items():
        key=Path(key).resolve().as_posix()
        if key in pins:assert pins[key]==value,'Conflicting review input pin'
        pins[key]=value
    assert pins.get(Path(path).resolve().as_posix())==digest,'Actual source review must be an explicit input'


'''
    text=replace(text,'def fresh():',helper+'def fresh():')
    put(BASE/'prepare_launch.py',text)
    text=(OLD/'test_launch.py').read_text().replace('independent_plant_pending_result_saved_actual_v1','independent_plant_timing_saved_actual_v1').replace('independent_plant_pending_result_saved_audit_v1','independent_plant_timing_saved_audit_v1')
    text=replace(text,'from prepare_launch import render','from prepare_launch import render,validate_source_review,SOURCE_PREPARATION_SHA,BASE\nimport copy')
    newtest='''    def test_actual_source_review_and_full_map_required(self):
        path=BASE/'synthetic_review.json';source_map={str(i)+'.py':'a'*64 for i in range(25)}
        review=dict(passed=True,source_review_pass=True,source_preparation_sha256=SOURCE_PREPARATION_SHA,source_sha256=source_map)
        q=dict(source_review={'path':str(path),'sha256':'b'*64},source_sha256=source_map,input_sha256={str(path):'b'*64})
        validate_source_review(q,path,'b'*64,review,source_map)
        for key,value in [('passed',False),('source_review_pass',False),('source_preparation_sha256','c'*64),('source_sha256',{})]:
            bad=copy.deepcopy(review);bad[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):validate_source_review(q,path,'b'*64,bad,source_map)
        for key,value in [('input_sha256',{}),('source_review',{'path':str(path)+'wrong','sha256':'b'*64})]:
            bad=copy.deepcopy(q);bad[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):validate_source_review(bad,path,'b'*64,review,source_map)

'''
    text=replace(text,"    def test_fixed_retry_v2_source_command(self):",newtest+"    def test_fixed_retry_v2_source_command(self):")
    put(BASE/'test_launch.py',text)

if __name__=='__main__':main()
