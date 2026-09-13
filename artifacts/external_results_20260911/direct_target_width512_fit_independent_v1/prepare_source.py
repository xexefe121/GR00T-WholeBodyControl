"""Derive saved-only width audit from preserved, completed corrected warm audit."""
from pathlib import Path
import hashlib,json,difflib
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_response_balanced_fit_independent_v3'
SOURCE=BASE/'source_prepared_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
SOURCE.mkdir(exist_ok=True)
prep=read(OLD/'source_preparation.json');changes={};unchanged={}
def replace(text,old,new):
    assert old in text,old
    return text.replace(old,new)
for name,digest in prep['source_sha256'].items():
    origin=OLD/'source_prepared_v1'/name;assert sha(origin)==digest
    text=origin.read_text();before=text
    if name=='audit_saved_warm.py':
        pairs=[('Audit saved warm71000 evidence.','Audit saved width81000 evidence.'),
            ('moments,rate,drift,metrics','moments,drift,metrics'),
            ('energy_weights,balanced_metrics,warm_initial_errors,ledger_expectations','energy_weights,balanced_metrics,ledger_expectations'),
            ('from audit_release import release_paths','from audit_release import release_paths\nfrom audit_width_math import rate,width_initial_errors,restoration_fields'),
            ('saved_response_balanced_warm_only','saved_width512_warm_only'),
            ('training_forward_rows=44058000,training_forward_calls=9000,training_updates=3000','training_forward_rows=146860000,training_forward_calls=30000,training_updates=10000'),
            ("kind='causal_response_balanced_continuation'","kind='causal_width512_warm_continuation'"),
            ('updates=3000,ordinary_start_step=68000,ordinary_final_step=71000,optimizer_start_step=3000,optimizer_final_step=6000','updates=10000,ordinary_start_step=71000,ordinary_final_step=81000,optimizer_start_step=6000,optimizer_final_step=16000'),
            ('budgets=expected_budget,learning_rate=[1e-5,1e-6]','budgets=expected_budget,learning_rate=dict(ramp_updates=250,ramp_inclusive=[1e-6,1e-5],cosine_updates=9750,cosine_inclusive=[1e-5,1e-6])'),
            ("'paths','full_state_paths','restoration_predictions','context_paths'","'paths','full_state_paths','restoration_predictions','context_paths','schedule_paths'"),
            ("'schedule_axes.npy','runtime.json'","'schedule_axes.npy','source_schedule_centers.npy','source_schedule_axes.npy','schedule_lineage.json','reused_inputs.json','runtime.json'"),
            ("name='source_output_manifest.json' if key=='manifest' else Path(path).name","name='source_output_manifest.json' if key=='manifest' else ('source_'+Path(path).name if key.startswith('schedule_') else Path(path).name)"),
            ("check('source68000',source['ordinary_final_step']==68000 and request['subjects']['checkpoint']['sha256']=='10d238198e2d7826c35eaa9bd4a13ff2e05050a7f601c1f212b585c4a90eaefd')","check('source71000',source['ordinary_final_step']==71000 and request['subjects']['checkpoint']['sha256']=='395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d')"),
            ("source_report['ordinary_final_step']==68000","source_report['ordinary_final_step']==71000"),
            ('ordinary_final_step=71000,additional_updates=3000,optimizer_step=6000','ordinary_final_step=81000,additional_updates=10000,optimizer_step=16000'),
            ('ordinary_start_step=68000,optimizer_start_step=3000','ordinary_start_step=71000,optimizer_start_step=6000'),
            ('split_contiguous_1000_plus_323','split_old256_new256_original1000_plus323'),
            ('shapes=((256,1323),(256,),(256,256),(256,),(23,256),(23,))','shapes=((512,1323),(512,),(512,512),(512,),(23,512),(23,))'),
            ("check('warm_initial_actor_optimizer_RNG',not warm_initial_errors(init,source))","check('width_initial_actor_optimizer_RNG',not width_initial_errors(init,source))\n            tree('retained_expansion_generator',checkpoint['expansion_generator_state'],init['expansion_generator_state'])"),
            ("float(state['step'])==6000","float(state['step'])==16000"),
            ("optimization['ordinary_final_step']==71000 and optimization['optimizer_step']==6000","optimization['ordinary_final_step']==81000 and optimization['optimizer_step']==16000"),
            ('progress.shape==(3000,6)','progress.shape==(10000,6)'),
            ('range(3000)],np.float64)','range(10000)],np.float64)'),
            ('cells.shape==(3000,width)','cells.shape==(10000,width)'),
            ('originals.shape==(3000,2)','originals.shape==(10000,2)'),
            ("('w0',(256,1323)),('b0',(256,)),('w1',(256,256)),('b1',(256,)),('w2',(23,256))","('w0',(512,1323)),('b0',(512,)),('w1',(512,512)),('b1',(512,)),('w2',(23,512))"),
            ('counts(9000,44058000)','counts(30000,146860000)')]
        for a,b in pairs:text=replace(text,a,b)
        text=replace(text,"initial_parity_tolerance_rad=1e-5,parity_tolerance_rad=1e-5).items()", "initial_parity_tolerance_rad=1e-5,parity_tolerance_rad=1e-5,features=1323,architecture=[1323,512,512,23],old_hidden_width=256,new_hidden_width=512,expansion_seed=20260912,context_order='previous_action23_then_incoming_history300',first_layer_execution='split_old256_new256_original1000_plus323',export_first_layer_execution='monolithic_float64_1323',automatic_retry=False,no_checkpoint_selection=True,context_and_normalization_reused=True,old_optimizer_blocks_exact=True,new_optimizer_moments_zero=True,shared_optimizer_step_for_new_entries=6000).items()")
        start=text.index('        sc,sa,_,_=schedule(3000)');stop=text.index('        data=dict(',start)
        text=text[:start]+'''        sc,sa,_,_=schedule(10000)
        for key,value in [('centers',sc),('axes',sa)]:
            old=npy(request['context_paths']['schedule_'+key]);check('old_schedule_shape:'+key,old.shape==(3000,864))
            exact('prior_schedule_prefix:'+key,old,value[:3000])
            exact('copied_prior_schedule:'+key,npy(shared/('source_schedule_'+key+'.npy')),old)
            exact('full_schedule_source:'+key,npy(request['schedule_paths']['schedule_'+key]),value)
            exact('shared_full_schedule:'+key,npy(shared/('schedule_'+key+'.npy')),value)
        compare('schedule_lineage',read(shared/'schedule_lineage.json'),dict(full_updates=10000,prior_updates=3000,prior_prefix_byte_exact=True,schedule_generated=False,source_centers_sha256=sha(request['schedule_paths']['schedule_centers']),source_axes_sha256=sha(request['schedule_paths']['schedule_axes']),prior_centers_sha256=sha(request['context_paths']['schedule_centers']),prior_axes_sha256=sha(request['context_paths']['schedule_axes'])))
        copied={('source_output_manifest.json' if key=='manifest' else ('source_'+Path(path).name if key.startswith('schedule_') else Path(path).name)):sha(path) for key,path in request['context_paths'].items()}
        compare('reused_inputs',read(shared/'reused_inputs.json'),dict(context_paths=request['context_paths'],schedule_paths=request['schedule_paths'],copied_sha256=copied,chronology_proof_sha256=request['subjects']['context_proof']['sha256'],context_reconstructed=False,normalization_recomputed=False,schedule_generated=False))
''' +text[stop:]
        start=text.index("            compare(condition+'_restoration',restoration,");stop=text.index("            check(condition+'_initial_drift_gate'",start)
        text=text[:start]+'''            compare(condition+'_restoration',restoration,dict(source_checkpoint_sha256=request['subjects']['checkpoint']['sha256'],**restoration_fields(),initial_drift=dict(passed=maximum<=1e-5,tolerance_rad=1e-5,max_preclip_error_rad=maximum,corpora=restored,source_ordinary_step=71000,changed_MatMul_dimension=True,stored_first_layer_width=1323,stored_hidden_width=512,old_block_contractions_preserved=True,first_layer_execution='split_old256_new256_original1000_plus323',byte_gate_required=False)))
''' +text[stop:]
        text=replace(text,"for field,value in dict(ordinary_start_step=71000,optimizer_start_step=6000,context_and_normalization_reused=True", "for field,value in dict(ordinary_start_step=71000,optimizer_start_step=6000,hidden_width=512,architecture=[1323,512,512,23],expansion_seed=20260912,context_and_normalization_reused=True")
        text=replace(text,"dict(ordinary_start_step=71000,optimizer_start_step=6000,condition=condition", "dict(ordinary_start_step=71000,optimizer_start_step=6000,hidden_width=512,expansion_seed=20260912,condition=condition")
        text=replace(text,"fresh_optimizer=False,context_order='previous_action23_then_incoming_history300'", "fresh_optimizer=False,hidden_width=512,architecture=[1323,512,512,23],expansion_seed=20260912,context_order='previous_action23_then_incoming_history300'")
        text=replace(text,"            for label,opt,lr in [('initial'", "            tree('final_whole_optimizer_group',optimizer['param_groups'],source['optimizer_state']['param_groups'])\n            for label,opt,lr in [('initial'")
        text=replace(text,"'Nominal float32 reductions use3e-7 relative tolerance;", "'Initialization checks reconstruct saved tensors with a local CPU generator; they do not execute a model. New moment entries retain shared age6000.',\n                'Nominal float32 reductions use3e-7 relative tolerance;")
    elif name=='audit_graph.py':
        text=replace(text,'[(256,1323),(256,256),(23,256)]','[(512,1323),(512,512),(23,512)]')
    elif name=='test_prior_math.py':
        for a,b in [('(256,1323)','(512,1323)'),('(256,256)','(512,512)'),('(23,256)','(23,512)'),('(256,)','(512,)')]:text=replace(text,a,b)
    elif name=='test_metadata.py':text=replace(text,'direct_target_causal_response_balanced_student_v2','direct_target_causal_width512_student_v1')
    destination=SOURCE/name
    if text==before:
        with destination.open('xb') as f:f.write(origin.read_bytes())
    else:
        with destination.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    if text==before:unchanged[name]=digest
    else:changes[name]=''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile='prior/'+name,tofile='width/'+name))
helpers={}
for name in ('prepare_audit_request.py','freeze_launch.py','run_audit_durable.ps1','verify_completion.py'):
    before=(OLD/name).read_text();text=before
    text=text.replace('saved_response_balanced_warm_only','saved_width512_warm_only').replace('direct_target_causal_response_balanced_student_v2','direct_target_causal_width512_student_v1')
    if name=='prepare_audit_request.py':text=replace(text,"report['ordinary_final_step']==71000 and report['optimizer_step']==6000","report['ordinary_final_step']==81000 and report['optimizer_step']==16000 and report['hidden_width']==512")
    if text==before:
        with (BASE/name).open('xb') as f:f.write((OLD/name).read_bytes())
    else:
        with (BASE/name).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    helpers[name]=dict(prior_sha256=sha(OLD/name),sha256=sha(BASE/name),text_unchanged=text==before)
    if text!=before:changes[name]=''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile='prior/'+name,tofile='width/'+name))
with (BASE/'source_delta.patch').open('x',encoding='utf-8') as f:f.write('\n'.join(changes.values()))
with (BASE/'derivation.json').open('x') as f:json.dump(dict(prior_preparation_sha256=sha(OLD/'source_preparation.json'),unchanged_source_sha256=unchanged,changed_files=list(changes),helpers=helpers,task_arrays_read=0,task_checkpoint_loads=0,model_calls=0,native_calls=0),f,indent=2)
