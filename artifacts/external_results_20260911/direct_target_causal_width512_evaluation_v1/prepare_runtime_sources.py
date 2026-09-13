"""Source-only adaptation for a future completed width512 endpoint."""
import difflib,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
NEW=BASE.parent
OLD=NEW/'direct_target_causal_response_evaluation_v1'
DEST=BASE/'source_draft_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def one(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
assert not DEST.exists()
prep=read(OLD/'source_preparation.json')
assert sha(OLD/'source_preparation.json')=='2195d4f5d06b6c4dec5d2a24038a179be0e266483fc0454edea540705d40fef3'
for name,h in prep['source_sha256'].items():
    source=OLD/'source_draft_v1'/name;assert sha(source)==h
    target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(source.read_bytes())

path=DEST/'evaluation_gate.py';s=path.read_text()
s=one(s,"binding['ordinary_final_step']==71000","binding['ordinary_final_step']==81000")
s=one(s,'direct_absolute_target_1323_causal_response_balanced','direct_absolute_target_1323_causal_width512')
s=one(s,"binding['architecture']==[1323,256,256,23]","binding['architecture']==[1323,512,512,23]")
path.write_text(s,encoding='utf-8',newline='\n')

path=DEST/'context_release.py';s=path.read_text()
s=one(s,'Literal single warm71000 release checks','Literal single width512 warm81000 release checks')
s=one(s,"SOURCE_CHECKPOINT='10d238198e2d7826c35eaa9bd4a13ff2e05050a7f601c1f212b585c4a90eaefd'","SOURCE_CHECKPOINT='395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d'")
s=one(s,'condition=condition,ordinary_final_step=71000','condition=condition,ordinary_final_step=81000')
s=one(s,'additional_updates=3000,optimizer_step=6000','additional_updates=10000,optimizer_step=16000')
s=one(s,'ordinary_start_step=68000,optimizer_start_step=3000,context_and_normalization_reused=True,',
      "ordinary_start_step=71000,optimizer_start_step=6000,context_and_normalization_reused=True,\n        architecture=[1323,512,512,23],hidden_width=512,expansion_seed=20260912,\n        training_first_layer_execution='split_old256_new256_original1000_plus323',export_first_layer_execution='monolithic_float64_1323',")
s=one(s,"counts['training']==counter(9000,44058000)","counts['training']==counter(30000,146860000)")
s=one(s,'ordinary71000_warm_balanced_context_same_weight_fp64_export','ordinary81000_width512_warm_balanced_context_same_weight_fp64_export')
start=s.index('    expected_fields(request,dict(')
end=s.index("    assert request['subjects']['checkpoint']['sha256']==SOURCE_CHECKPOINT",start)
block=s[start:end]
block=one(block,"kind='causal_response_balanced_continuation'","kind='causal_width512_warm_continuation'")
block=one(block,'updates=3000,ordinary_start_step=68000,ordinary_final_step=71000,optimizer_start_step=3000,optimizer_final_step=6000,',
               'updates=10000,ordinary_start_step=71000,ordinary_final_step=81000,optimizer_start_step=6000,optimizer_final_step=16000,')
block=one(block,'learning_rate=[1e-5,1e-6],features=1323,',
      "learning_rate=dict(ramp_updates=250,ramp_inclusive=[1e-6,1e-5],cosine_updates=9750,cosine_inclusive=[1e-5,1e-6]),features=1323,\n        architecture=[1323,512,512,23],old_hidden_width=256,new_hidden_width=512,expansion_seed=20260912,\n        old_optimizer_blocks_exact=True,new_optimizer_moments_zero=True,shared_optimizer_step_for_new_entries=6000,\n        first_layer_execution='split_old256_new256_original1000_plus323',export_first_layer_execution='monolithic_float64_1323',")
s=s[:start]+'    training_request_identity(request)\n'+s[end:]
where=s.index('\ndef validate_release(')
s=s[:where]+'\ndef training_request_identity(request):\n'+block+'\n'+s[where:]
path.write_text(s,encoding='utf-8',newline='\n')

path=DEST/'test_context_release.py';s=path.read_text()
s=one(s,'condition=condition,ordinary_final_step=71000','condition=condition,ordinary_final_step=81000')
s=one(s,'additional_updates=3000,optimizer_step=6000','additional_updates=10000,optimizer_step=16000')
s=one(s,'ordinary_start_step=68000,optimizer_start_step=3000,context_and_normalization_reused=True,',
      "ordinary_start_step=71000,optimizer_start_step=6000,context_and_normalization_reused=True,\n        architecture=[1323,512,512,23],hidden_width=512,expansion_seed=20260912,\n        training_first_layer_execution='split_old256_new256_original1000_plus323',export_first_layer_execution='monolithic_float64_1323',")
s=one(s,'training=counter(9000,44058000)','training=counter(30000,146860000)')
s=one(s,"('condition','blinded'),('ordinary_final_step',68000),('features',1000),", "('condition','blinded'),('ordinary_final_step',71000),('features',1000),\n    ('architecture',[1323,256,256,23]),('hidden_width',256),('expansion_seed',0),('additional_updates',3000),\n    ('training_first_layer_execution','monolithic'),('export_first_layer_execution','float32'),")
path.write_text(s,encoding='utf-8',newline='\n')

changed={n:sha(DEST/n) for n in prep['source_sha256'] if sha(DEST/n)!=prep['source_sha256'][n]}
assert set(changed)=={'evaluation_gate.py','context_release.py','test_context_release.py'}
diff=''.join(''.join(difflib.unified_diff((OLD/'source_draft_v1'/n).read_text().splitlines(True),(DEST/n).read_text().splitlines(True),fromfile='preserved71000/'+n,tofile='width81000/'+n)) for n in changed)
with (BASE/'source_changes.diff').open('x',encoding='utf-8') as f:f.write(diff)
result=dict(preparation_only=True,old_source_preparation_sha256=sha(OLD/'source_preparation.json'),
 original_source_sha256=prep['source_sha256'],changed_source_sha256=changed,unchanged_sources=34,
 source_sha256={n:sha(DEST/n) for n in prep['source_sha256']},source_changes_sha256=sha(BASE/'source_changes.diff'),
 actual_model_calls=0,native_steps=0,actual_binding_created=False,writer_sha256=sha(Path(__file__)))
with (BASE/'runtime_derivation.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'derived_sources':37,'unchanged_sources':34,'runtime_derivation_sha256':sha(BASE/'runtime_derivation.json')}))
