"""Deterministic source-only derivation; never imports task runtime modules."""
from pathlib import Path

BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_causal_context_study_v2/source_snapshot_v1'
DEST=BASE/'source_draft_v1'
text=(OLD/'train_context_pair.py').read_text(encoding='utf-8')
def change(old,new,count=1):
    global text
    if text.count(old)!=count:raise ValueError('Unexpected derivation occurrence: '+old[:90])
    text=text.replace(old,new)

change('"""Future selected pair only; immutable data/objective, no adaptive fitting."""',
       '"""One cleared warm causal continuation; fixed teacher-energy response weighting."""')
change('from full_state_objective import full_state_loss','from balanced_response_objective import balanced_full_state_loss')
change('from context_contract import CONDITIONS,UPDATES,COEFFICIENT,BUDGETS,cosine_rate,fixed_schedule',
       'from response_contract import UPDATES,COEFFICIENT,BUDGETS,cosine_rate')
change('from context_data import load_context_data,condition_data','from response_data import load_response_data')
change('from context_model import ContextTarget,expand_actor,assert_blinded_columns','from context_model import ContextTarget')
change('from context_promoted import from_checkpoint,export_onnx','from response_promoted import from_checkpoint,export_onnx')
change('from context_support import gate','from response_support import gate,final_response_identity\nfrom response_restore import check_warm_source,verify_warm_restoration\nfrom response_diagnostics import summarize_balanced,save_balanced_prefix\nfrom balance_contract import GROUP_WEIGHTS,WEIGHT_RULE')
change("Saved65000 prediction schema.","Saved causal68000 prediction schema.")
change('Nonfinite initial dimensional drift.','Nonfinite restored initial drift.')
change("changed_MatMul_dimension=False,stored_first_layer_width=1323", "source_ordinary_step=68000,changed_MatMul_dimension=False,stored_first_layer_width=1323")
change('def run_condition(condition,data,source,expanded,normalization,sc,sa,request,receipt,identity,shared):\n    dest=BASE/\'fit\'/condition;dest.mkdir(exist_ok=False)',
       "def run_continuation(data,source,normalization,sc,sa,request,receipt,identity,shared):\n    condition='causal';dest=BASE/'fit'")
change("attempted_global_step=65000", "attempted_global_step=68000")
change('losses=[];nc_records=[];fc_records=[];pc_records=[];', 'losses=[];nc_records=[];fc_records=[];pc_records=[];original_losses=[];weighted_cell_records=[];')
change('model.actor.load_state_dict(expanded,strict=True);model=model.to(\'cuda\')',"model.actor.load_state_dict(source['actor_state'],strict=True);model=model.to('cuda')")
change("        restore_rng(source['rng']);rng=rng_save()\n        exact_saved(model.actor.state_dict(),expanded,'expanded initial actor');exact_saved(rng,source['rng'],'restored65000 RNG')\n        if optimizer.state:raise ValueError('Fresh AdamW must be empty.')",
       "        optimizer.load_state_dict(copy.deepcopy(source['optimizer_state']))\n        restore_rng(source['rng']);rng=rng_save()\n        restoration=verify_warm_restoration(model,optimizer,source,rng)")
change('ordinary_start_step=65000,optimizer_start_step=0,','ordinary_start_step=68000,optimizer_start_step=3000,')
start=text.index("        write(dest/'restoration.json',")
end=text.index("        state['stage']='training_tensor_setup'",start)
text=text[:start]+"""        write(dest/'restoration.json',dict(source_checkpoint_sha256=request['subjects']['checkpoint']['sha256'],
            **restoration,initial_drift=drift))
        if not drift['passed']:raise ValueError('Restored initial preclamp drift exceeds1e-5; no updates.')
"""+text[end:]
change('attempted_global_step=65001+index,attempted_optimizer_step=index+1,','attempted_global_step=68001+index,attempted_optimizer_step=3001+index,')
change('            fl,fc=full_state_loss(nominal,full,tensor(mapped),fy[tensor(rows),tensor(axes)])',
       '            balanced=balanced_full_state_loss(nominal,full,tensor(mapped),fy[tensor(rows),tensor(axes)])\n            fl=balanced.weighted_objective;fc=balanced.original_cells\n            original_fl=balanced.original_objective;weighted_fc=balanced.weighted_cells')
change('active.update(nominal_loss=nl.detach(),full_state_loss=fl.detach(),physical_loss=pl.detach(),total_loss=loss.detach(),',
       'active.update(nominal_loss=nl.detach(),balanced_full_state_loss=fl.detach(),physical_loss=pl.detach(),balanced_total_loss=loss.detach(),')
change('                nominal_cell_losses=nc.detach(),full_state_cell_losses=fc.detach(),physical_cell_losses=pc.detach())',
       '                nominal_cell_losses=nc.detach(),full_state_cell_losses=fc.detach(),physical_cell_losses=pc.detach(),\n                original_full_state_loss=original_fl.detach(),balanced_full_state_cell_losses=weighted_fc.detach())')
change('            nc_records.append(nc.detach().cpu().numpy().copy());fc_records.append(fc.detach().cpu().numpy().copy());pc_records.append(pc.detach().cpu().numpy().copy())',
       '            nc_records.append(nc.detach().cpu().numpy().copy());fc_records.append(fc.detach().cpu().numpy().copy());pc_records.append(pc.detach().cpu().numpy().copy())\n            original_losses.append([float(original_fl.detach()),float((nl+COEFFICIENT*original_fl+pl).detach())])\n            weighted_cell_records.append(weighted_fc.detach().cpu().numpy().copy())')
change('save_loss_prefix(dest,losses,nc_records,fc_records,pc_records)',
       'save_balanced_prefix(dest,losses,nc_records,fc_records,pc_records,original_losses,weighted_cell_records)',3)
change('ordinary_step=65001+index','ordinary_step=68001+index')
change("any(int(v['step'])!=3000 for v in optimizer.state.values())", "any(int(v['step'])!=6000 for v in optimizer.state.values())")
change("        if condition=='blinded':assert_blinded_columns(model.actor.state_dict())\n",'')
change('ordinary_final_step=68000','ordinary_final_step=71000',4)
change('optimizer_step=3000','optimizer_step=6000',3)
change('fresh_optimizer=True','fresh_optimizer=False',2)
change("context_blinded=condition=='blinded'", "context_blinded=False")
change('full_state_coefficient=COEFFICIENT,request=request',"full_state_coefficient=COEFFICIENT,response_group_weights=GROUP_WEIGHTS,response_weight_rule=WEIGHT_RULE,request=request")
change('del x,fx,px,y,fy,py,cells,pcells,successors,nominal,full,physical,nl,fl,pl,loss,nc,fc,pc',
       'del x,fx,px,y,fy,py,cells,pcells,successors,nominal,full,physical,nl,fl,pl,loss,nc,fc,pc,balanced,original_fl,weighted_fc')
change("            metrics[label]=summarize(outputs[label],data)\n            metrics[label]['weighted_objective']=metrics[label]['nominal_objective']+COEFFICIENT*metrics[label]['full_state_objective']+metrics[label]['physical_objective']",
       '            metrics[label]=summarize_balanced(outputs[label],data)')
change('final_identity(BASE,identity,receipt)','final_response_identity(BASE,identity,receipt)',2)
change("            head_output='normalized_target',fixed_full_state_coefficient=COEFFICIENT,counts=counts,metrics=metrics,",
       "            head_output='normalized_target',fixed_full_state_coefficient=COEFFICIENT,counts=counts,metrics=metrics,\n            response_group_weights=GROUP_WEIGHTS,response_weight_rule=WEIGHT_RULE,\n            ordinary_start_step=68000,optimizer_start_step=3000,context_and_normalization_reused=True,\n            initial_restoration=drift,training_loss_columns=['nominal','balanced_full_state','physical','balanced_total','learning_rate','preclip_gradient_norm'],\n            source_checkpoint_sha256=request['subjects']['checkpoint']['sha256'],\n            energy_source_sha256=request['subjects']['energy_source']['sha256'],\n            direct_subject_sha256={key:value['sha256'] for key,value in request['subjects'].items()},")
change('Endpoint same-weight FP64 parity failed; pair incomplete.','Endpoint same-weight FP64 parity failed; continuation incomplete.')
change("        failure=dict(error=repr(exc),state=state,counts=counts,preservation_errors=errors,optimization_completed=optimized,",
       "        final_check=False\n        try:final_response_identity(BASE,identity,receipt);final_check=True\n        except BaseException as error:errors.append('final identity: '+repr(error))\n        failure=dict(error=repr(exc),state=state,counts=counts,preservation_errors=errors,optimization_completed=optimized,\n            all_frozen_inputs_unchanged=final_check,")
text=text[:text.index('\ndef main():')]+'''
def main():
    import shutil
    receipt,request,identity=gate(BASE,__file__)
    dest=BASE/'fit';dest.mkdir(exist_ok=False);shared=dest/'shared';shared.mkdir()
    stage='load_reused_data';run_started=False
    try:
        data,normalization,sc,sa=load_response_data(request,receipt['input_sha256'])
        source=torch.load(request['subjects']['checkpoint']['path'],map_location='cpu',weights_only=True)
        check_warm_source(source)
        for key in ('feature_mean','feature_std'):
            exact_saved(source[key],torch.from_numpy(normalization[key]),'saved full1323 '+key)
        for key,value in [('joint_span',data['span']),('runtime_joint_span',data['span'].astype(np.float64)),('default_q',data['default']),('joint_limits',data['limits']),('retained_feature_indices',RETAINED)]:
            exact_saved(torch.from_numpy(value),source[key],key)
        # Preserve the qualified files byte-for-byte; no context normalization or schedule generation.
        copied={}
        for key,path in request['context_paths'].items():
            name='source_output_manifest.json' if key=='manifest' else Path(path).name
            shutil.copyfile(path,shared/name);copied[name]=sha(shared/name)
            if copied[name]!=sha(path):raise ValueError('Copied reused context bytes differ: '+key)
        write(shared/'reused_inputs.json',dict(context_paths=request['context_paths'],copied_sha256=copied,
            chronology_proof_sha256=request['subjects']['context_proof']['sha256'],
            context_reconstructed=False,normalization_recomputed=False,schedule_generated=False))
        write(shared/'runtime.json',configure_runtime(request))
        write(shared/'output_manifest.json',dict(files=manifest(shared),training_request_sha256=identity['request'],frozen_receipt_sha256=identity['frozen']))
        stage='run_continuation';run_started=True
        report=run_continuation(data,source,normalization,sc,sa,request,receipt,identity,shared)
        print('CONTINUATION_COMPLETE',sha(dest/'report.json'),flush=True)
    except BaseException as exc:
        if not run_started:
            unchanged=False;identity_error=None
            try:final_response_identity(BASE,identity,receipt);unchanged=True
            except BaseException as error:identity_error=repr(error)
            failure=dict(completed=False,optimization_completed=False,stage=stage,error=repr(exc),
                task_forward_calls=0,optimizer_updates=0,all_frozen_inputs_unchanged=unchanged,
                final_identity_error=identity_error,automatic_retry=False)
            write(dest/'setup_failure.json',failure);write(dest/'report.json',failure)
        raise

if __name__=='__main__':main()
'''
(DEST/'train_response_balanced.py').write_text(text,encoding='utf-8',newline='\n')
promoted=(OLD/'context_promoted.py').read_text(encoding='utf-8')
promoted=promoted.replace('68000','71000').replace('paired ordinary','warm ordinary').replace('Paired context architecture','Causal context architecture')
(DEST/'response_promoted.py').write_text(promoted,encoding='utf-8',newline='\n')
print('Source derivation written; no task runtime imported or executed.')
