"""Source text derivation only. No numerical imports, data or checkpoint loads."""
from pathlib import Path
import hashlib
import json
import difflib

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
OLD=NEW/'direct_target_causal_response_balanced_student_v2/source_prepared_v1'
OUT=BASE/'source_draft_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def change(text,old,new,count=1):
    if text.count(old)!=count:raise ValueError('Source derivation count differs: '+repr(old))
    return text.replace(old,new)

OUT.mkdir(exist_ok=False)
for path in OLD.glob('*.py'):
    with (OUT/path.name).open('xb') as f:f.write(path.read_bytes())
width=NEW/'direct_target_causal_width512_preparation_v1/source_prepared_v1/width512.py'
with (OUT/'width512.py').open('xb') as f:f.write(width.read_bytes())

path=OUT/'train_response_balanced.py';original=path.read_text();text=original
text=change(text,'One cleared warm causal continuation; fixed teacher-energy response weighting.',
            'One reviewed width512 warm continuation; unchanged teacher-energy response objective.')
text=change(text,'from response_contract import UPDATES,COEFFICIENT,BUDGETS,cosine_rate',
            'from response_contract import UPDATES,COEFFICIENT,BUDGETS,cosine_rate,START_STEP,FINAL_STEP,OPTIMIZER_START,OPTIMIZER_FINAL')
text=change(text,'from context_model import ContextTarget','from width512 import WiderContextTarget,expand_source,validate_source')
text=change(text,'from response_promoted import from_checkpoint,export_onnx','from width512_promoted import from_checkpoint,export_onnx')
text=change(text,'from response_restore import check_warm_source,verify_warm_restoration','from width_restore import verify_width_restoration')
text=text.replace('causal68000','causal71000').replace('68001+index','START_STEP+1+index')
text=text.replace('attempted_global_step=68000','attempted_global_step=START_STEP')
text=text.replace('source_ordinary_step=68000,changed_MatMul_dimension=False,stored_first_layer_width=1323,first_layer_execution=\'split_contiguous_1000_plus_323\'',
                  "source_ordinary_step=71000,changed_MatMul_dimension=True,stored_first_layer_width=1323,stored_hidden_width=512,old_block_contractions_preserved=True,first_layer_execution='split_old256_new256_original1000_plus323'")
old="""model=ContextTarget(normalization['feature_mean'],normalization['feature_std'])
        model.actor.load_state_dict(source['actor_state'],strict=True);model=model.to('cuda')"""
new="""expansion=expand_source(source)
        model=WiderContextTarget(normalization['feature_mean'],normalization['feature_std'])
        model.actor.load_state_dict(expansion['actor_state'],strict=True);model=model.to('cuda')"""
text=change(text,old,new)
text=change(text,"optimizer.load_state_dict(copy.deepcopy(source['optimizer_state']))","optimizer.load_state_dict(copy.deepcopy(expansion['optimizer_state']))")
text=change(text,'restoration=verify_warm_restoration(model,optimizer,source,rng)',
            'restoration=verify_width_restoration(model,optimizer,source,expansion,rng)')
text=change(text,'rng_after_restoration=rng,condition=condition,ordinary_start_step=68000,optimizer_start_step=3000,',
            "rng_after_restoration=rng,expansion_seed=expansion['expansion_seed'],expansion_generator_state=expansion['expansion_generator_state'],\n            condition=condition,ordinary_start_step=START_STEP,optimizer_start_step=OPTIMIZER_START,hidden_width=512,")
text=change(text,"shape=(3000,864)","shape=(UPDATES,864)",count=2)
text=change(text,'attempted_optimizer_step=3001+index','attempted_optimizer_step=OPTIMIZER_START+1+index')
text=text.replace('9000,44058000',"BUDGETS['training_forward_calls'],BUDGETS['training_forward_rows']")
text=change(text,"any(int(v['step'])!=6000", "any(int(v['step'])!=OPTIMIZER_FINAL")
text=text.replace('ordinary_final_step=71000,additional_updates=3000,optimizer_step=6000',
                  'ordinary_final_step=FINAL_STEP,additional_updates=UPDATES,optimizer_step=OPTIMIZER_FINAL')
text=change(text,'ordinary_final_step=71000,optimizer_step=6000',
            'ordinary_final_step=FINAL_STEP,optimizer_step=OPTIMIZER_FINAL')
text=change(text,'ordinary_start_step=68000,optimizer_start_step=3000,context_and_normalization_reused=True,',
            'ordinary_start_step=START_STEP,optimizer_start_step=OPTIMIZER_START,context_and_normalization_reused=True,\n            architecture=[1323,512,512,23],hidden_width=512,expansion_seed=expansion[\'expansion_seed\'],')
text=change(text,"training_first_layer_execution='split_contiguous_1000_plus_323'", "training_first_layer_execution='split_old256_new256_original1000_plus323'")
text=change(text,"fresh_optimizer=False,actor_state=cpu_tree(model.actor.state_dict())", "fresh_optimizer=False,hidden_width=512,architecture=[1323,512,512,23],expansion_seed=expansion['expansion_seed'],\n            expansion_generator_state=expansion['expansion_generator_state'],actor_state=cpu_tree(model.actor.state_dict())")
text=change(text,'check_warm_source(source)','validate_source(source)')
text=change(text,"name='source_output_manifest.json' if key=='manifest' else Path(path).name", "name='source_output_manifest.json' if key=='manifest' else ('source_'+Path(path).name if key.startswith('schedule_') else Path(path).name)")
text=change(text,"write(shared/'reused_inputs.json',dict(context_paths=request['context_paths'],copied_sha256=copied,", "for key,path in request['schedule_paths'].items():\n            shutil.copyfile(path,shared/Path(path).name)\n            if sha(shared/Path(path).name)!=sha(path):raise ValueError('Copied full schedule differs: '+key)\n        write(shared/'schedule_lineage.json',data['schedule_lineage'])\n        write(shared/'reused_inputs.json',dict(context_paths=request['context_paths'],schedule_paths=request['schedule_paths'],copied_sha256=copied,")
path.write_text(text,encoding='utf-8',newline='\n')

path=OUT/'response_support.py';original_support=path.read_text();support=original_support
support=change(support,"('paths','full_state_paths','restoration_predictions','context_paths')", "('paths','full_state_paths','restoration_predictions','context_paths','schedule_paths')")
support=support.replace("ordinary_final_step']!=68000", "ordinary_final_step']!=71000").replace("optimizer_step']!=3000", "optimizer_step']!=6000").replace('causal68000','causal71000')
path.write_text(support,encoding='utf-8',newline='\n')

path=OUT/'response_data.py';original_data=path.read_text();data=original_data
needle="""    return condition_data(data,'causal'),normalization,sc,sa"""
replacement="""    full_centers_path=checked_file(request['schedule_paths']['schedule_centers'],pins)
    full_axes_path=checked_file(request['schedule_paths']['schedule_axes'],pins)
    full_centers=np.load(full_centers_path,allow_pickle=False)
    full_axes=np.load(full_axes_path,allow_pickle=False)
    verify_schedule(full_centers,full_axes,data['full_state_centers']['dataset'],data['full_state_centers']['control'],
                    data['full_state_centers']['axis_group'],updates=10000)
    exact(full_centers[:3000],sc,'original10000 schedule first3000 center bytes')
    exact(full_axes[:3000],sa,'original10000 schedule first3000 axis bytes')
    data['schedule_lineage']=dict(full_updates=10000,prior_updates=3000,prior_prefix_byte_exact=True,
        schedule_generated=False,source_centers_sha256=sha(full_centers_path),source_axes_sha256=sha(full_axes_path),
        prior_centers_sha256=sha(paths['schedule_centers']),prior_axes_sha256=sha(paths['schedule_axes']))
    return condition_data(data,'causal'),normalization,full_centers,full_axes"""
data=change(data,needle,replacement);path.write_text(data,encoding='utf-8',newline='\n')
derivation={}
patch=[]
for path in OUT.glob('*.py'):
    prior=OLD/path.name
    if prior.exists():
        derivation[path.name]=dict(old_sha256=sha(prior),new_sha256=sha(path),byte_identical=prior.read_bytes()==path.read_bytes())
        patch.extend(difflib.unified_diff(prior.read_text().splitlines(True),path.read_text().splitlines(True),fromfile='old/'+path.name,tofile='new/'+path.name))
with (BASE/'source_derivation_initial.json').open('x',encoding='utf-8') as f:json.dump(derivation,f,indent=2);f.write('\n')
with (BASE/'source_derivation_initial.patch').open('x',encoding='utf-8') as f:f.write(''.join(patch))
print(json.dumps(dict(source_files=len(list(OUT.glob('*.py'))),source_only=True,model_calls=0,native_steps=0)))
