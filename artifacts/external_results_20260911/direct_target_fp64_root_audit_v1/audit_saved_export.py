"""Saved-only independent audit of the selected same-weight FP64 export."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_graph import exact, audit_graph
from audit_arrays import measure_outputs, CORPORA, NEW
from audit_math import summarize

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def run(experiment, output):
    experiment, output = Path(experiment), Path(output)
    output.mkdir(exist_ok=False)
    export = experiment/'export'
    tracked = {}
    checks = []
    result = dict(passed=False, evidence_audit_passed=False)
    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None: assert digest == expected, str(path)
        if str(path) in tracked: assert tracked[str(path)] == digest, str(path)+' changed'
        tracked[str(path)] = digest
        return path
    def read(path): return json.loads(bind(path).read_text(encoding='utf-8-sig'))
    def role(item): return bind(item['path'],item['sha256'])
    def check(name, condition):
        checks.append(dict(name=name,passed=bool(condition)))
        assert condition, name
    def compare(a,b,name):
        if isinstance(b,dict):
            check(name+'.keys',set(a)==set(b))
            for key in b: compare(a[key],b[key],name+'.'+key)
        elif isinstance(b,list):
            check(name+'.length',len(a)==len(b))
            for i,v in enumerate(b): compare(a[i],v,name+'['+str(i)+']')
        elif isinstance(b,float):
            tolerance=3e-7 if name.endswith(('normalized_MSE','nominal_objective')) else 5e-12
            check(name,bool(np.isclose(a,b,rtol=tolerance,atol=1e-14,equal_nan=False)))
        else: check(name,a==b)
    def archive(path):
        with np.load(bind(path),allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
    try:
        for name in ('audit_saved_export.py','audit_graph.py','audit_arrays.py','audit_math.py'):
            bind(Path(__file__).with_name(name))
        request_path=experiment/'export_request.json'
        frozen_path=experiment/'export_frozen_inputs.json'
        clearance_path=experiment/'export_clearance.json'
        request, frozen, clearance = read(request_path), read(frozen_path), read(clearance_path)
        check('selected_scope',request['kind']=='one_same_weight_fp64_export_validation' and request['root_selected'] is True)
        check('fixed_parameters',request['ordinary_final_step']==55000 and request['features']==1000 and request['batch_size']==256 and request['parity_tolerance_rad']==1e-5)
        check('fixed_partitions',request['backend_order']==list(NEW) and request['corpus_rows']==list(CORPORA.values()) and request['per_backend_batches']==[39,550,12])
        compare(request['budgets'],dict(Torch_calls=1202,Torch_rows=307160,ORT_calls=601,ORT_rows=153580,trace_forward_calls=0,optimizer_updates=0,BFM_calls=0,native_steps=0),'fixed_budgets')
        check('no_retry_or_new_checkpoint',request['automatic_retry'] is False and request['checkpoint_selection'] is False and request['canonical_evaluation_authorized'] is False)
        check('request_frozen_identity',frozen['export_request_sha256']==sha(request_path))
        check('approved_actual_subjects',clearance['approved'] is True and clearance['request_sha256']==sha(request_path) and clearance['frozen_receipt_sha256']==sha(frozen_path))
        review=read(role(dict(path=clearance['review_path'],sha256=clearance['review_sha256'])))
        check('concrete_source_review',review['prelaunch_review_pass'] is True and review['export_request_sha256']==sha(request_path) and review['frozen_receipt_sha256']==sha(frozen_path))
        bind(clearance['launcher_path'],clearance['launcher_sha256'])
        frozen_pins={str(Path(p).resolve()):v for p,v in frozen['input_sha256'].items()}
        for path,digest in frozen_pins.items(): bind(path,digest)
        for name,digest in frozen['source_sha256'].items(): bind(Path(frozen['source_directory'])/name,digest)
        subjects={name:role(item) for name,item in request['subjects'].items()}
        for name,path in subjects.items(): check('consumed_subject.'+name,frozen_pins[str(path)]==sha(path))
        for name,path in request['paths'].items():
            # The original loader's configuration has only file-role paths.
            path=Path(path).resolve();bind(path,frozen_pins[str(path)])
        for backend,corpora in request['old_outputs'].items():
            for corpus,path in corpora.items():
                path=Path(path).resolve();bind(path,frozen_pins[str(path)])
                check('fixed_old_output.'+backend+'.'+corpus,path==subjects['checkpoint'].parent/('final_'+backend+'_'+corpus+'.npy'))
        training=read(subjects['training_review']);training_audit=read(subjects['training_audit'])
        check('verified_training_only',training['training_review_pass'] is True and training['dataset_review_pass'] is True and training['export_review_pass'] is False)
        check('training_audit_preserves_failed_release',training_audit['evidence_audit_passed'] is True and training_audit['export_qualified'] is False and training_audit['canonical_cleared'] is False)
        check('training_review_binds_audit',training['direct_subject_sha256']['root_training_audit']==sha(subjects['training_audit']))
        for name in ('checkpoint','fit_report','source_head','normalization','training_request'):
            check('training_review_subject.'+name,training['direct_subject_sha256'][name]==sha(subjects[name]))
        original=read(subjects['fit_report'])
        check('original_failure_preserved',original['completed'] is False and original['numerical_gate_passed'] is False and original['export_parity_passed'] is False and original['optimization_completed'] is True)
        check('exact_ordinary_checkpoint',sha(subjects['checkpoint'])=='9ceef5099ebd154e08a1c2c3784c4e06021d464e607474548763665b24f8f9e7')
        norm=archive(subjects['normalization'])
        import torch, onnx
        checkpoint=torch.load(subjects['checkpoint'],map_location='cpu',weights_only=True)
        check('ordinary55000_metadata',checkpoint['ordinary_final_step']==55000 and checkpoint['additional_updates']==50000 and checkpoint['kind']=='direct_absolute_native23_target')
        graph_path=bind(export/'student_head_fp64.onnx')
        graph_result=audit_graph(onnx.load(graph_path),checkpoint,norm)
        promoted=archive(export/'promoted_parameters.npz')
        check('promoted_archive_keys',set(promoted)=={'feature_mean','feature_std','w0','b0','w1','b1','w2','b2'})
        for name in ('feature_mean','feature_std'): exact(promoted[name],checkpoint[name].numpy().astype(np.float64),name)
        for layer,index in enumerate((0,2,4)):
            for dest,source in [('w','weight'),('b','bias')]:
                exact(promoted[dest+str(layer)],checkpoint['actor_state'][str(index)+'.'+source].numpy().astype(np.float64),'promoted '+dest)
        promotion=read(export/'promotion.json')
        check('promotion_receipt',promotion['passed'] is True and promotion['checkpoint_sha256']==sha(subjects['checkpoint']) and promotion['all_source_f32_values_exactly_promoted'] is True and promotion['trainable_parameters']==promotion['optimizer_updates']==0)
        numerical, outputs=measure_outputs(export,subjects['checkpoint'].parent,norm,bind)
        paths=request['paths'];teachers=[]
        for key in ('centers','pico','walk002'):
            with np.load(bind(paths[key]),allow_pickle=False) as z: teachers.append(z['expert_target'].copy())
        physical_manifest=read(paths['physical_manifest']);physical_dir=Path(paths['physical_manifest']).parent
        def physical(key):
            item=physical_manifest['arrays'][key]
            return np.load(bind(physical_dir/item['path'],item['sha256']),allow_pickle=False)
        data=dict(default=norm['default_q'],span=norm['joint_span'],limits=norm['joint_limits'],
            nominal_teacher=np.concatenate(teachers),velocity_teacher=np.load(bind(paths['velocity_target']),allow_pickle=False).reshape(140622,23),
            physical_teacher=physical('label_fixed_map_target'),physical_successor=physical('successor_center_index'),
            physical_flags={key:physical(key) for key in ('policy_native_target_clipped','teacher_feedback_clipped','teacher_native_clipped','successor_replan_boundary','successor_zero_gain')})
        metrics={}
        for backend in NEW:
            o=outputs[backend]
            metrics[backend]=summarize(o['nominal'],o['velocity'],o['physical'],data)
            compare(read(export/('metrics_'+backend+'.json')),metrics[backend],'metrics.'+backend)
        manifest=read(export/'manifest.json')
        check('completed_output_manifest',manifest['completed'] is True)
        expected_files={p.name for p in export.iterdir() if p.is_file() and p.name not in ('manifest.json','report.json','progress.json') and p.suffix!='.tmp'}
        check('complete_output_inventory',set(manifest['files'])==expected_files)
        for name,item in manifest['files'].items():
            check('manifest_basename.'+name,item['path']==name and Path(name).name==name)
            p=bind(export/name,item['sha256'])
            if p.suffix=='.npy':
                value=np.load(p,mmap_mode='r',allow_pickle=False)
                check('manifest_array_schema.'+name,list(value.shape)==item['shape'] and str(value.dtype)==item['dtype'])
        report=read(export/'report.json');qualified=numerical['parity_passed']
        check('report_numerical_classification',report['validation_completed'] is True and report['completed'] is qualified and report['numerical_gate_passed'] is qualified and report['export_parity_passed'] is qualified)
        check('report_parity_exact',report['max_preclip_error_rad']==numerical['maximum_preclamp_rad'] and report['parity_tolerance_rad']==1e-5)
        compare(report['counters'],numerical['counters'],'report_counters')
        for field,key in [('checkpoint_sha256','checkpoint'),('source_fit_report_sha256','fit_report'),('source_onnx_sha256','source_head'),('normalization_sha256','normalization')]:
            check('report_subject.'+key,report[field]==sha(subjects[key]))
        for field,path in [('onnx_sha256',graph_path),('export_request_sha256',request_path),('frozen_receipt_sha256',frozen_path),('manifest_sha256',export/'manifest.json'),('graph_calls_sha256',export/'graph_calls.jsonl')]:
            check('report_subject.'+field,report[field]==sha(path))
        for field in ('optimizer_updates','BFM_calls','native_steps','trace_forward_calls'): check('zero_'+field,report[field]==0)
        check('report_same_mathematical_model',report['ordinary_final_step']==55000 and report['features']==1000 and report['head_output']=='normalized_target' and report['execution_dtype']=='float64' and report['public_input_dtype']==report['public_output_dtype']=='float32' and report['ELU_implementation']=='Where(x>0,x,Exp(Min(x,0))-1)')
        check('report_unchanged_inputs',report['all_frozen_inputs_unchanged'] is True and report['checkpoint_selection'] is False)
        for path,digest in tracked.items(): check('final_hash:'+path,sha(path)==digest)
        direct={name:sha(subjects[name]) for name in ('checkpoint','fit_report','source_head','normalization')}
        direct.update(head=sha(graph_path),export_report=sha(export/'report.json'),export_request=sha(request_path),export_manifest=sha(export/'manifest.json'),root_training_audit=sha(subjects['training_audit']))
        result=dict(passed=qualified,evidence_audit_passed=True,export_qualified=qualified,canonical_evaluation_cleared=False,
            checks=len(checks),graph=graph_result,numerical=numerical,metrics=metrics,direct_subject_sha256=direct,input_sha256=tracked,
            original_FP32_export_qualified=False,same_trained_weights=True,new_FP64_execution_semantics=True,
            task_model_calls=0,ORT_calls=0,native_steps=0,optimizer_updates=0,
            limitations=['Saved evidence and literal graph audit do not establish closed-loop stability or runtime timing.','Producer owner/process completion and final export review are separate mandatory release gates.'])
    except BaseException as exc:
        result=dict(passed=False,evidence_audit_passed=False,error=repr(exc),checks=len(checks),input_sha256=tracked,
                    task_model_calls=0,ORT_calls=0,native_steps=0,optimizer_updates=0)
        raise
    finally:
        (output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=result['passed'],evidence_audit_passed=result['evidence_audit_passed'],checks=len(checks),report_sha256=sha(output/'report.json'))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.experiment,a.output)
