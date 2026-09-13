"""Pure saved-data context proof; no Torch/checkpoint load, model or native calls."""
from pathlib import Path
import json
import numpy as np
from direct_data import read,sha
from context_data import load_context_data,condition_data
from context_contract import fixed_schedule
from full_state_contract import verify_schedule

BASE=Path(__file__).resolve().parent.parent

def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def main():
    proposal=read(BASE/'training_request_proposal.json');pins=read(BASE/'preparation_inputs.json')
    if proposal['root_selected'] is not False or pins['actual_execution_authorized'] is not False:raise ValueError('Preparation-only proof required.')
    if sha(BASE/'training_request_proposal.json')!=pins['training_request_proposal_sha256']:raise ValueError('Preparation request changed.')
    source_pins={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    dest=BASE/'context_preflight';dest.mkdir(exist_ok=False)
    data=None
    try:
        data=load_context_data(proposal,pins['input_sha256'])
        for key in ('nominal_context','center_context','physical_context'):np.save(dest/(key+'.npy'),data[key])
        np.savez_compressed(dest/'context_normalization.npz',context_mean=data['context_mean'],context_std=data['context_std'],
            context_mean64=data['context_mean64'],context_variance64=data['context_variance64'])
        old_centers=np.load(proposal['schedule_paths']['centers'],allow_pickle=False);old_axes=np.load(proposal['schedule_paths']['axes'],allow_pickle=False)
        verify_schedule(old_centers,old_axes,data['full_state_centers']['dataset'],data['full_state_centers']['control'],data['full_state_centers']['axis_group'])
        centers,axes=fixed_schedule(old_centers,old_axes)
        np.save(dest/'schedule_centers.npy',centers);np.save(dest/'schedule_axes.npy',axes)
        feature_proof={}
        for condition in ('blinded','causal'):
            expanded=condition_data(data,condition)
            for key in ('features','full_state_features','physical_features'):
                # Prove every expanded row keeps exact current1000 and exact selected context.
                count=0
                for start in range(0,len(data[key]),1024):
                    value=expanded[key][start:start+1024]
                    assert value[:,:1000].tobytes()==data[key][start:start+1024].tobytes()
                    if condition=='blinded':assert np.all((value[:,1000:]-data['context_mean'])/data['context_std']==0)
                    count+=len(value)
                feature_proof[condition+'_'+key]=count
        for path,digest in pins['input_sha256'].items():assert sha(path)==digest,path
        for name,digest in source_pins.items():assert sha(Path(__file__).parent/name)==digest,name
        output={p.name:sha(p) for p in dest.iterdir() if p.is_file()}
        report=dict(passed=True,proof=data['context_proof'],feature_proof=feature_proof,
            original_feature_uniqueness_retained_by_exact1000_prefix=True,
            input_sha256=pins['input_sha256'],source_sha256=source_pins,output_sha256=output,
            proposal_sha256=sha(BASE/'training_request_proposal.json'),all_inputs_unchanged=True,
            model_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0)
        write_new(dest/'report.json',report)
        print(json.dumps(dict(report_sha256=sha(dest/'report.json'),nominal_pairs=data['context_proof']['nominal_chronological_history_and_prior_pairs'],
            physical_prior_raw_difference_rows=data['context_proof']['physical_applied_prior_differs_raw_rows'],model_calls=0,native_steps=0)))
    except BaseException as exc:
        if data is not None:
            for key in ('nominal_context','center_context','physical_context'):
                if key in data and not (dest/(key+'.npy')).exists():np.save(dest/(key+'.npy'),data[key])
        write_new(dest/'failure.json',dict(passed=False,error=repr(exc),proposal_sha256=sha(BASE/'training_request_proposal.json'),source_sha256=source_pins,
            existing_output_sha256={p.name:sha(p) for p in dest.iterdir() if p.is_file()},model_calls=0,native_steps=0,automatic_retry=False))
        raise

if __name__=='__main__':main()
