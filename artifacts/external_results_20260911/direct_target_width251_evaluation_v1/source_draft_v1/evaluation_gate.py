"""No model creation until a separately selected real final export is bound."""
import re
from pathlib import Path
import sys
import json
import hashlib
import numpy as np

CENTERS_SHA='5b07595d07e262f2ad236565ec62483d600843fe27a9f4a781974dac4b0608c7'
LABEL_SHA='bdcba927cd737a2ba74d75d7b31c427a5bfd4df7437983d13c391f9ca166f749'
INDICES=np.r_[np.arange(52),np.arange(75,1023)].astype(np.int64)


def local(value):
    text=str(value).replace('\\','/')
    if sys.platform!='win32' and len(text)>2 and text[1]==':':text='/mnt/'+text[0].lower()+text[2:]
    return Path(text)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def exact(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def field(value,path):
    for key in path.split('.'):value=value[key]
    return value
def has_hash(value,digest):
    if isinstance(value,dict):return any(has_hash(v,digest) for v in value.values())
    if isinstance(value,list):return any(has_hash(v,digest) for v in value)
    return value==digest
def bound_file(entry):
    if not isinstance(entry,dict) or re.fullmatch('[0-9a-f]{64}',str(entry.get('sha256',''))) is None:
        raise ValueError('actual literal SHA256 and existing absolute artifact required')
    path=local(entry['path'])
    if not path.is_absolute() or not path.is_file() or sha(path)!=entry['sha256']:
        raise ValueError('bound artifact absent or changed: '+str(path))
    return path


def require_model_ready(base,purpose):
    if purpose not in ('witness','evaluation'):raise ValueError('invalid execution purpose')
    base=Path(base);path=base/(purpose+'_binding.json')
    if not path.is_file():raise ValueError('Source preparation is unbound; no model/native call permitted.')
    binding=read(path)
    assert binding['root_authorized_'+purpose] is True
    assert binding['ordinary_final_step']==91000
    assert binding['controller']=='direct_absolute_target_1323_causal_width512_recovery'
    assert binding['architecture']==[1323,512,512,23]
    assert binding['context_condition']=='causal'
    assert binding['hidden_activation']=='ELU' and binding['head_output']=='normalized_target'
    assert binding['span_contract']=='existing_float32_joint_span_promoted_float64'
    assert binding['learned_BFM_calls']==0 and binding['hardware_authorized'] is False
    assert binding['clock_foundation_connected'] is False
    assert binding['filters_enabled'] is False and binding['compiled_preview_enabled'] is False
    if purpose=='witness':assert binding['expected_head_calls']==1 and binding['physics_authorized'] is False
    else:assert binding['requested_main_controls']==1569 and binding['conditional_hold_controls']==250
    names=('head','checkpoint','fit_report','training_manifest','training_request','centers','query250_labels','contract',
           'normalization','export_manifest','coefficient','source_checkpoint','full_state_generation_request',
           'full_state_generation_report','full_state_data_audit','full_state_data_owner',
           'shared_manifest','context_alignment','energy_source',
           'recovery_rows','collection_report','collection_request','collection_qualification',
           'collection_source_review','consistency_report','warm_restore_review')
    paths={name:bound_file(binding[name]) for name in names}
    assert binding['centers']['sha256']==CENTERS_SHA and binding['query250_labels']['sha256']==LABEL_SHA
    for entry in binding['input_files']:bound_file(entry)
    # Every actual source module is pinned in the execution binding.
    source=Path(__file__).resolve().parent
    entries={local(item['path']).resolve():item['sha256'] for item in binding['input_files']}
    for file in source.rglob('*.py'):
        assert file.resolve() in entries and sha(file)==entries[file.resolve()],str(file)
    from context_release import validate_release
    fit=validate_release(binding,paths,source,purpose,read=read,sha=sha,bound_file=bound_file,field=field,has_hash=has_hash)
    with np.load(paths['centers'],allow_pickle=False) as z:
        assert z['dataset'][2038]==2 and z['control'][2038]==250 and z['source_frame'][2038]==261
        center={key:z[key][2038].copy() for key in ('features','qpos','qvel','state','history','previous_action')}
        span=z['joint_span'].copy();limits=z['joint_limits'].copy()
    with np.load(paths['query250_labels'],allow_pickle=False) as z:
        assert z['control'][0]==250 and z['source_frame'][0]==261
        for key in ('features','state','history','previous_action'):assert exact(center[key],z[key][0]),key
        assert exact(center['qpos'],z['teacher_qpos'][0]) and exact(center['qvel'],z['teacher_qvel'][0])
        assert exact(span,z['joint_span']) and exact(limits,z['joint_limits'])
    assert center['features'].dtype==np.float32 and center['features'].shape==(1069,)
    current=center['features'][INDICES].copy()
    with np.load(paths['normalization'],allow_pickle=False) as z:
        context_mean=z['context_mean'].copy()
        assert context_mean.shape==(323,) and context_mean.dtype==np.float32 and np.isfinite(context_mean).all()
        assert z['feature_mean'].shape==(1323,) and z['feature_std'].shape==(1323,)
        assert exact(z['feature_mean'][1000:],context_mean)
    actual_context=np.concatenate((center['previous_action'],center['history']))
    assert actual_context.shape==(323,) and actual_context.dtype==np.float32 and np.isfinite(actual_context).all()
    selected_context=actual_context if binding['context_condition']=='causal' else context_mean
    center['current_features']=current.copy()
    center['actual_context']=actual_context.copy()
    center['features']=np.concatenate((current,selected_context))
    c=read(paths['contract']);default=np.asarray(c['default_q'],np.float64)
    assert exact(limits,np.asarray(c['joint_limits'],np.float64))
    assert span.dtype==np.float32 and span.shape==(23,)
    assert exact(span,np.diff(limits,axis=1)[:,0].astype(np.float32))
    return dict(binding=binding,binding_path=path,binding_sha256=sha(path),paths=paths,fit_report=fit,
                head=paths['head'],center=center,span=span,default=default,limits=limits,
                context_condition=binding['context_condition'],context_mean=context_mean)


def require_ready(base):
    ready=require_model_ready(base,'evaluation');b=ready['binding']
    paths={key:bound_file(b[key]) for key in ('first_export_witness','first_export_receipt')}
    report=read(paths['first_export_receipt'])
    assert report['pass_all'] is True and report['expected_head_calls']==report['attempted_head_calls']==report['returned_head_calls']==1
    assert report['BFM_inference_calls']==report['physics_steps']==0
    assert report['head_sha256']==b['head']['sha256'] and report['witness_sha256']==b['first_export_witness']['sha256']
    assert report['features']==1323 and report['context_condition']==b['context_condition']
    assert report['exact_extended_query250_features'] is True
    assert report['onnxruntime_version']=='1.23.2' and report['pinned_WSL_runtime'] is True
    assert report['intra_op_threads']==report['inter_op_threads']==1 and report['execution_mode']=='ORT_SEQUENTIAL'
    assert report['execution_provider']=='CPUExecutionProvider'
    assert any(entry['sha256']==report['runtime_binary_sha256'] for entry in b['input_files'])
    with np.load(paths['first_export_witness'],allow_pickle=False) as z:export={key:z[key].copy() for key in z.files}
    for key in ('features','span','default','limits'):assert exact(export[key],ready['center']['features'] if key=='features' else ready[key]),key
    assert str(export['context_condition'].item())==b['context_condition']
    assert exact(export['actual_context'],ready['center']['actual_context'])
    assert export['normalized_target'].shape==(23,) and export['normalized_target'].dtype==np.float32
    assert np.isfinite(export['normalized_target']).all()
    raw=ready['default']+ready['span'].astype(np.float64)*export['normalized_target'].astype(np.float64)
    assert exact(export['raw_proposal'],raw)
    assert exact(export['target'],np.clip(raw,ready['limits'][:,0],ready['limits'][:,1]))
    assert str(export['head_sha256'].item())==b['head']['sha256']
    assert str(export['runtime_binary_sha256'].item())==report['runtime_binary_sha256']
    ready['first_export']=export
    return ready
