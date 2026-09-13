"""Fresh saved-only81000 adaptation; never run after source freeze."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
OLD=NEW/'direct_target_response_saved_semantics_review_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def one(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)
def main():
    prep=json.loads((OLD/'source_preparation.json').read_text())
    for name,digest in prep['source_sha256'].items():assert sha(OLD/name)==digest,name
    for name in prep['source_sha256']:
        if name in ('derive_sources.py','derive_helpers.py','record_preparation.py'):continue
        original=(OLD/name).read_text();text=original
        if name=='saved_common.py':text=one(text,'direct_target_causal_response_evaluation_v1','direct_target_causal_width512_evaluation_v1')
        elif name=='release_checks.py':
            for a,b in [('causal71000','causal81000'),('ordinary71000_warm_balanced_context_same_weight_fp64_export','ordinary81000_width512_warm_balanced_context_same_weight_fp64_export'),('[1323,256,256,23]','[1323,512,512,23]'),("==71000","==81000"),('71000 causal release kind','81000 causal width512 release kind'),('completed71000','completed81000'),("==68000","==71000"),("==3000","==10000"),("fit['optimizer_start_step']==10000","fit['optimizer_start_step']==6000"),("fit['optimizer_step']==6000","fit['optimizer_step']==16000"),('10d238198e2d7826c35eaa9bd4a13ff2e05050a7f601c1f212b585c4a90eaefd','395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d'),('actual causal68000 source','actual causal71000 source')]:
                assert a in text,a;text=text.replace(a,b)
            marker="    check.require(fit['fixed_full_state_coefficient']"
            at=text.index(marker)
            text=text[:at]+"    check.require(fit['architecture']==[1323,512,512,23] and fit['hidden_width']==512 and fit['expansion_seed']==20260912,'selected expanded width and seed')\n    check.require(fit['training_first_layer_execution']=='split_old256_new256_original1000_plus323' and fit['export_first_layer_execution']=='monolithic_float64_1323','split training and qualified dense FP64 export')\n"+text[at:]
        elif name=='prepare_request.py':
            text=text.replace("==71000","==81000").replace('one_pure_saved_causal71000_runtime_semantics_and_fixed_maps','one_pure_saved_causal81000_width512_runtime_semantics_and_fixed_maps')
            text=one(text,"binding['ordinary_final_step']==81000","binding['ordinary_final_step']==81000 and binding['architecture']==[1323,512,512,23]")
        elif name=='prepare_audit_stage.py':text=one(text,'8b11bcdb646779089ab1118468b8753b06317619455e02e8896aaaceb97742a3','6a6f771154e3cbbc7660713f77a12df6592fcf39f0962c34739c58101cfc7f1e')
        elif name=='test_audit_stage.py':text=one(text,'direct_target_response_saved_semantics_review_v1','direct_target_width512_saved_semantics_review_v1')
        elif name=='test_context_math.py':
            text=text.replace('[1323,256,256,23]','[1323,512,512,23]').replace('ordinary_final_step=71000','ordinary_final_step=81000').replace('ordinary71000_warm_balanced_context_same_weight_fp64_export','ordinary81000_width512_warm_balanced_context_same_weight_fp64_export')
        elif name=='test_response_release.py':
            text=text.replace('ordinary_start_step=68000','ordinary_start_step=71000').replace('ordinary_final_step=71000','ordinary_final_step=81000').replace('additional_updates=3000,optimizer_start_step=3000,optimizer_step=6000','additional_updates=10000,optimizer_start_step=6000,optimizer_step=16000').replace('10d238198e2d7826c35eaa9bd4a13ff2e05050a7f601c1f212b585c4a90eaefd','395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d')
            text=one(text,'fresh_optimizer=False,fixed_full_state_coefficient=',"architecture=[1323,512,512,23],hidden_width=512,expansion_seed=20260912,training_first_layer_execution='split_old256_new256_original1000_plus323',export_first_layer_execution='monolithic_float64_1323',fresh_optimizer=False,fixed_full_state_coefficient=")
            text=text.replace('direct_target_context_saved_semantics_review_v1','direct_target_response_saved_semantics_review_v1')
        if text==original:
            with (BASE/name).open('xb') as f:f.write((OLD/name).read_bytes())
        else:
            with (BASE/name).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    with (BASE/'preserved_run_audit_template.ps1.txt').open('xb') as f:f.write((OLD/'preserved_run_audit_template.ps1.txt').read_bytes())
if __name__=='__main__':main()
