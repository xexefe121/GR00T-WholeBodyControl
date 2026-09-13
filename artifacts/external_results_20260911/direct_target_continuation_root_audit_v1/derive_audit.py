"""Prepare continuation auditor from reviewed 5000-step auditor; no fit reads."""
from pathlib import Path
import ast
import hashlib
import json

B=Path(__file__).resolve().parent
OLD=B.parent/'direct_target_root_audit_v1'
source=(OLD/'audit_saved_fit.py').read_text()
changes=[]
def change(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old))
    source=source.replace(old,new);changes.append({'old':old,'new':new})

change('from audit_math import balanced_moments, phase_indices, summarize, targets','from audit_math import balanced_moments, phase_indices, summarize, targets\nfrom audit_restoration import differences')
change("sources=[Path(__file__),Path(__file__).with_name('audit_math.py')]","sources=[Path(__file__),Path(__file__).with_name('audit_math.py'),Path(__file__).with_name('audit_restoration.py')]")
change("report['ordinary_final_step']==5000","report['ordinary_final_step']==55000")
change("exact('all_2880000_sampled_centers',npy(fit/'replayed_center_rows.npy'),npy(paths['sampled_rows']))","exact('all_28800000_repeated_sampled_centers',npy(fit/'replayed_center_rows.npy'),np.tile(npy(paths['sampled_rows']),(10,1)))")
change("exact('all_2880000_sampled_axes',npy(fit/'replayed_axes.npy'),npy(paths['sampled_axes']))","exact('all_28800000_repeated_sampled_axes',npy(fit/'replayed_axes.npy'),np.tile(npy(paths['sampled_axes']),(10,1)))")
change("check('5000_complete_finite_loss_records',progress.shape==(5000,6)","check('50000_complete_finite_loss_records',progress.shape==(50000,6)")
change('for index in range(5000):','for index in range(50000):')
change('rate=3e-5+.5*(3e-4-3e-5)*(1+math.cos(math.pi*index/4999))','rate=3e-6+.5*(3e-5-3e-6)*(1+math.cos(math.pi*index/49999))')
change("nc.shape==(5000,15) and vc.shape==pc.shape==(5000,9)","nc.shape==(50000,15) and vc.shape==pc.shape==(50000,9)")
change("'training_head_rows_attempted':70550000,'training_head_rows_returned':70550000","'training_head_rows_attempted':705500000,'training_head_rows_returned':705500000")
change("checkpoint['ordinary_final_step']==checkpoint['additional_updates']==5000","checkpoint['ordinary_final_step']==55000 and checkpoint['additional_updates']==50000")
change("optimization['ordinary_final_step']==5000","optimization['ordinary_final_step']==55000")
change("float(state['step'])==5000","float(state['step'])==55000")
change("group['lr']==3e-5","group['lr']==3e-6")
change("        compare('checkpoint_request',checkpoint['request'],request)","""        compare('checkpoint_request',checkpoint['request'],request)
        old_fit=experiment.parent/'direct_target_student_v1/fit'
        old_checkpoint_path=bind(old_fit/'student_head.pt')
        check('exact_selected_source_checkpoint',sha(old_checkpoint_path)=='ad6d657affba0796e7313d85ace240cd31d46e188c577da049880a7a031aecba')
        old_checkpoint=torch.load(old_checkpoint_path,map_location='cpu',weights_only=True)
        restored=torch.load(bind(fit/'initialization.pt'),map_location='cpu',weights_only=True)
        check('restoration_source_subject',restored['source_checkpoint_sha256']==sha(old_checkpoint_path))
        for name in ('actor_state','optimizer_state'):
            problems=differences(restored[name],old_checkpoint[name],name)
            check('restored_start_exact.'+name,not problems,problems)
        problems=differences(restored['rng_after_restoration'],old_checkpoint['rng'],'rng')
        check('restored_all_rng_exact',not problems,problems)
        problems=differences(checkpoint['rng'],old_checkpoint['rng'],'final_rng')
        check('no_new_rng_draws_in_fixed_continuation',not problems,problems)
        old_norm=archive(old_fit/'normalization.npz')
        check('normalization_keys_unchanged',set(old_norm)==set(norm))
        for key in norm:exact('source_normalization_exact.'+key,norm[key],old_norm[key])
        for corpus in ('nominal','velocity','physical'):
            exact('initial_GPU_matches_old_final_GPU.'+corpus,outputs['initial_GPU'][corpus],npy(old_fit/f'final_GPU_{corpus}.npy'))
        compare('initial_metrics_match_old_final',read(bind(fit/'initial_metrics.json')),read(bind(old_fit/'final_metrics.json')))
        check('selected_additional_budget',request['updates']==50000)
""")
change("'RNG initialization and CUDA execution flags covered by trainer source/runtime review, not reproduced by this array audit'","'restored model/optimizer/RNG and initial predictions are checked exactly; CUDA execution flags covered by trainer source/runtime review'")
ast.parse(source)
out=B/'audit_saved_fit.py'
assert not out.exists()
out.write_text(source)
for name in ('audit_math.py','test_audit_math.py'):
    assert not (B/name).exists()
    (B/name).write_bytes((OLD/name).read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(B/'source_derivation.json').write_text(json.dumps(dict(original_sha256=sha(OLD/'audit_saved_fit.py'),source_sha256=sha(out),changes=changes,
    unchanged_math_sha256=sha(B/'audit_math.py'),model_calls=0,optimizer_calls=0,native_steps=0),indent=2)+'\n')
print(json.dumps(dict(source_sha256=sha(out),changes=len(changes))))
