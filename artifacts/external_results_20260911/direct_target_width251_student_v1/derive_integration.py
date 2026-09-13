"""Create a source-only candidate by explicit checked text substitutions."""
from pathlib import Path
import difflib

BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_causal_width512_student_v1/source_prepared_v1'
DEST=BASE/'source_draft_v1'

def replace(source,old,new,count=1):
    if source.count(old)!=count:raise ValueError('Unexpected source occurrence: '+old[:100])
    return source.replace(old,new)

def write(name,source):
    with (DEST/name).open('x',encoding='utf-8',newline='\n') as f:f.write(source)

def main():
    diagnostics=(OLD/'context_diagnostics.py').read_text()
    for before,after in [
        ("CORPORA = ('nominal','full_state','physical')","CORPORA = ('nominal','full_state','physical','recovery')"),
        ('SIZES = (9904,354612,3054)','SIZES = (9904,354612,3054,1018)'),
        ("FEATURE_KEYS = ('features','full_state_features','physical_features')","FEATURE_KEYS = ('features','full_state_features','physical_features','recovery_features')"),
        ('CALLS = 1437','CALLS = 1441')]:diagnostics=replace(diagnostics,before,after)
    write('recovery_diagnostics.py',diagnostics)
    promoted=(OLD/'width512_promoted.py').read_text()
    promoted=replace(promoted,'def from_checkpoint(saved):','def from_checkpoint(saved,expected_step):\n    if type(expected_step) is not int or not 81000<expected_step<=91000:raise ValueError("Selected ordinary endpoint required")')
    promoted=replace(promoted,"saved['ordinary_final_step']!=81000","saved['ordinary_final_step']!=expected_step")
    promoted=replace(promoted,'Exact proposed ordinary81000 subject required','Exact selected ordinary recovery endpoint required')
    promoted=replace(promoted,'same_81000_width512_context_weights_fp64_execution','selected_recovery_width512_context_weights_fp64_execution')
    write('recovery_promoted.py',promoted)
    support=(OLD/'response_support.py').read_text()
    support=replace(support,'from response_contract import check_protocol,COEFFICIENT','from recovery_protocol import check_protocol,COEFFICIENT\nfrom recovery_data import check_recovery_subjects')
    support=replace(support,"source_report['ordinary_final_step']!=71000","source_report['ordinary_final_step']!=81000")
    support=replace(support,"source_report['optimizer_step']!=6000","source_report['optimizer_step']!=16000")
    support=replace(support,'Exact completed causal71000 fit required.','Exact completed causal81000 fit required.')
    support=replace(support,'    return receipt,request,identity','    check_recovery_subjects(request,receipt["input_sha256"])\n    return receipt,request,identity')
    write('recovery_support.py',support)
    original=(OLD/'train_response_balanced.py').read_text();s=original
    s=replace(s,'"""One reviewed width512 warm continuation; unchanged teacher-energy response objective."""','"""Conditional D3 integration draft; no fit without future literal reviewed protocol."""')
    s=replace(s,'from response_contract import UPDATES,COEFFICIENT,BUDGETS,cosine_rate,START_STEP,FINAL_STEP,OPTIMIZER_START,OPTIMIZER_FINAL',
        'from recovery_protocol import COEFFICIENT,START_STEP,OPTIMIZER_START,check_protocol')
    s=replace(s,'from response_data import load_response_data','from recovery_data import load_data\nfrom recovery_objective import recovery_loss')
    s=replace(s,'from width512 import WiderContextTarget,expand_source,validate_source','from width512 import WiderContextTarget\nfrom warm512_restore import restore as restore_warm512,validate_source')
    s=replace(s,'from context_diagnostics import BACKENDS,CORPORA,run_backend,summarize,parity,save_drift,atomic',
        'from context_diagnostics import CORPORA as OLD_CORPORA\nfrom recovery_diagnostics import BACKENDS,CORPORA,run_backend,parity,save_drift,atomic')
    s=replace(s,'from width512_promoted import from_checkpoint,export_onnx','from recovery_promoted import from_checkpoint,export_onnx')
    s=replace(s,'from response_support import gate,final_response_identity','from recovery_support import gate,final_response_identity')
    s=replace(s,'from width_restore import verify_width_restoration\nfrom response_diagnostics import summarize_balanced,save_balanced_prefix',
        'from recovery_metrics import summarize_recovery,save_prefix')
    s=replace(s,'    for corpus in CORPORA:\n','    for corpus in OLD_CORPORA:\n')
    s=replace(s,'Saved causal71000 prediction schema.','Saved causal81000 prediction schema.')
    s=replace(s,'source_ordinary_step=71000,changed_MatMul_dimension=True','source_ordinary_step=81000,changed_MatMul_dimension=False')
    s=replace(s,"    condition='causal';dest=BASE/'fit'","    protocol=check_protocol(request);UPDATES=protocol.updates;FINAL_STEP=protocol.final_step;OPTIMIZER_FINAL=protocol.optimizer_final;BUDGETS=protocol.budgets\n    condition='causal';dest=BASE/'fit'")
    s=replace(s,'losses=[];nc_records=[];fc_records=[];pc_records=[];original_losses=[];weighted_cell_records=[];',
        'losses=[];nc_records=[];fc_records=[];pc_records=[];original_losses=[];weighted_cell_records=[];recovery_losses=[];recovery_cells=[];')
    before="""        expansion=expand_source(source)
        model=WiderContextTarget(normalization['feature_mean'],normalization['feature_std'])
        model.actor.load_state_dict(expansion['actor_state'],strict=True);model=model.to('cuda')
        if len(list(model.parameters()))!=6 or any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Six float32 parameter tensors required.')
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-5,foreach=False,fused=False)
        optimizer.load_state_dict(copy.deepcopy(expansion['optimizer_state']))
        restore_rng(source['rng']);rng=rng_save()
        restoration=verify_width_restoration(model,optimizer,source,expansion,rng)"""
    after="""        model=WiderContextTarget(normalization['feature_mean'],normalization['feature_std']).to('cuda')
        optimizer=torch.optim.AdamW(model.actor.parameters(),lr=1e-6,weight_decay=1e-5,foreach=False,fused=False)
        state['restoration_progress']={}
        restoration=restore_warm512(model,optimizer,source,rng_save,state['restoration_progress']);rng=rng_save()"""
    s=replace(s,before,after)
    s=replace(s,"rng_after_restoration=rng,expansion_seed=expansion['expansion_seed'],expansion_generator_state=expansion['expansion_generator_state'],",
        'rng_after_restoration=rng,expansion_performed=False,')
    s=replace(s,"        x=tensor(data['features']);fx=tensor(data['full_state_features']);px=tensor(data['physical_features'])",
        "        x=tensor(data['features']);fx=tensor(data['full_state_features']);px=tensor(data['physical_features']);rx=tensor(data['recovery_features'])\n        ry=tensor(normalized_labels(data['recovery_target'],data['default'],data['span']));rcells=[tensor(ids) for ids in data['recovery_cells']]")
    s=replace(s,'rate=cosine_rate(index)','rate=protocol.rate(index)')
    before="            physical=counted_forward(model,px,'physical',counts['training'],active,BUDGETS['training_forward_calls'],BUDGETS['training_forward_rows'])"
    s=replace(s,before,before+"\n            recovery=counted_forward(model,rx,'recovery',counts['training'],active,BUDGETS['training_forward_calls'],BUDGETS['training_forward_rows'])")
    s=replace(s,'            loss=nl+COEFFICIENT*fl+pl',
        '            rl,rc=recovery_loss(recovery,ry,rcells)\n            old_total=nl+COEFFICIENT*fl+pl\n            loss=old_total+protocol.recovery_coefficient*rl')
    s=replace(s,'balanced_total_loss=loss.detach(),','balanced_total_loss=old_total.detach(),\n                recovery_loss=rl.detach(),recovery_cell_losses=rc.detach(),combined_recovery_loss=loss.detach(),')
    s=replace(s,'for v in (nl,fl,pl,loss)','for v in (nl,fl,pl,rl,loss)')
    s=replace(s,'float(pl.detach()),float(loss.detach()),rate','float(pl.detach()),float(old_total.detach()),rate')
    s=replace(s,'            weighted_cell_records.append(weighted_fc.detach().cpu().numpy().copy())',
        '            weighted_cell_records.append(weighted_fc.detach().cpu().numpy().copy())\n            recovery_losses.append([float(rl.detach()),float((protocol.recovery_coefficient*rl).detach()),float(loss.detach())])\n            recovery_cells.append(rc.detach().cpu().numpy().copy())')
    s=replace(s,'save_balanced_prefix(dest,losses,nc_records,fc_records,pc_records,original_losses,weighted_cell_records)',
        'save_prefix(dest,losses,nc_records,fc_records,pc_records,original_losses,weighted_cell_records,recovery_losses,recovery_cells)',count=3)
    s=replace(s,"fresh_optimizer=False,hidden_width=512,architecture=[1323,512,512,23],expansion_seed=expansion['expansion_seed'],\n            expansion_generator_state=expansion['expansion_generator_state'],actor_state=",
        "fresh_optimizer=False,hidden_width=512,architecture=[1323,512,512,23],expansion_performed=False,\n            source_expansion_seed=source['expansion_seed'],source_expansion_generator_state=source['expansion_generator_state'],actor_state=")
    s=replace(s,'full_state_coefficient=COEFFICIENT,response_group_weights=GROUP_WEIGHTS,',
        'full_state_coefficient=COEFFICIENT,recovery_coefficient=protocol.recovery_coefficient,response_group_weights=GROUP_WEIGHTS,')
    s=replace(s,'del x,fx,px,y,fy,py,cells,pcells,successors,nominal,full,physical,nl,fl,pl,loss,nc,fc,pc,balanced,original_fl,weighted_fc',
        'del x,fx,px,rx,y,fy,py,ry,cells,pcells,rcells,successors,nominal,full,physical,recovery,nl,fl,pl,rl,loss,nc,fc,pc,rc,balanced,original_fl,weighted_fc,old_total')
    s=replace(s,'promoted=from_checkpoint(checkpoint).eval()','promoted=from_checkpoint(checkpoint,FINAL_STEP).eval()')
    s=replace(s,'metrics[label]=summarize_balanced(outputs[label],data)','metrics[label]=summarize_recovery(outputs[label],data,protocol.recovery_coefficient)')
    s=replace(s,'expected_counter(1437,367570)','expected_counter(1441,368588)')
    s=replace(s,"architecture=[1323,512,512,23],hidden_width=512,expansion_seed=expansion['expansion_seed'],",
        "architecture=[1323,512,512,23],hidden_width=512,expansion_performed=False,\n            recovery_rows=1018,recovery_phase_counts=[99,819,100],recovery_coefficient=protocol.recovery_coefficient,\n            recovery_loss_columns=['recovery','weighted_recovery','combined_total'],recovery_collection=request['subjects']['collection_report'],")
    s=replace(s,"data,normalization,sc,sa=load_response_data(request,receipt['input_sha256'])","data,normalization,sc,sa=load_data(request,receipt['input_sha256'])")
    s=replace(s,"        write(shared/'schedule_lineage.json',data['schedule_lineage'])",
        "        shutil.copyfile(request['subjects']['recovery_rows']['path'],shared/'recovery_rows.npz')\n        if sha(shared/'recovery_rows.npz')!=request['subjects']['recovery_rows']['sha256']:raise ValueError('Copied recovery rows differ')\n        write(shared/'recovery_evidence.json',data['recovery_evidence'])\n        write(shared/'schedule_lineage.json',data['schedule_lineage'])")
    if 'expansion[' in s or 'expand_source(' in s or 'verify_width_restoration(' in s:raise ValueError('Legacy expansion leaked into new main')
    write('train_recovery.py',s)
    with (BASE/'main_derivation.patch').open('x',encoding='utf-8') as f:
        f.writelines(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='qualified_width81000/train_response_balanced.py',tofile='draft/train_recovery.py'))
    print('Derived four source modules; no task execution')

if __name__=='__main__':main()
