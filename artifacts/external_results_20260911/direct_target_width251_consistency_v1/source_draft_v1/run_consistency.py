"""Gated one-pass-mode saved consistency diagnosis. No model/native imports."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from admission import admit,local,read,sha
from saved_inputs import assemble
from consistency_math import MODES,OLD,TOTAL,BLOCK,alias_groups,fixed_queries,proximity
from normalization_math import verify_export_algebra

def save_json(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--clearance',type=Path,required=True);parser.add_argument('--clearance-sha256',required=True);args=parser.parse_args()
    source=Path(__file__).resolve().parent
    request=admit(args.request,args.clearance,args.clearance_sha256,source)
    if np.__version__!='1.26.4':raise RuntimeError('Actual saved diagnosis requires frozen NumPy1.26.4')
    output=local(request['output']);output.mkdir(exist_ok=False)
    progress=dict(stage='admitted',completed_alias_modes=[],proximity_queries_completed=0,model_calls=0,native_steps=0,optimizer_updates=0)
    try:
        subjects=request['subjects'];paths={k:local(v['path']) for k,v in subjects.items()}
        algebra=verify_export_algebra(paths['export_source'].read_text())
        physical={k:local(v['path']) for k,v in request['physical_arrays'].items()}
        x,y,labels,norm,metadata=assemble(paths,physical,read)
        assert len(x)==TOTAL
        np.savez_compressed(output/'rows.npz',features=x,**metadata)
        summaries={}
        for mode in MODES:
            progress['stage']='alias_'+mode
            arrays,summary=alias_groups(x,y,labels,norm['feature_mean'],norm['feature_std'],mode)
            np.savez_compressed(output/('aliases_'+mode+'.npz'),**arrays)
            summaries[mode]=summary;progress['completed_alias_modes'].append(mode)
            save_json(output/('completed_'+mode+'.json'),summary)
            del arrays
        queries=fixed_queries(metadata['phase'],metadata['control'])
        np.save(output/'proximity_queries.npy',queries)
        progress['stage']='proximity'
        # Each completed query is saved before moving to the next; interruption
        # cannot turn the absence of later evidence into a completed diagnosis.
        with (output/'proximity.jsonl').open('x',encoding='utf-8') as stream:
            for query in queries:
                records=proximity(x,y,metadata['phase'],[query],norm['feature_mean'],norm['feature_std'])
                for row in records:
                    for prefix,index in [('query',row['query_row']),('candidate',row['candidate_row'])]:
                        row[prefix+'_origin']=int(metadata['origin'][index]);row[prefix+'_origin_row']=int(metadata['origin_row'][index])
                        row[prefix+'_dataset']=int(metadata['dataset'][index]);row[prefix+'_phase']=int(metadata['phase'][index])
                        row[prefix+'_control']=int(metadata['control'][index]);row[prefix+'_source_frame']=int(metadata['source_frame'][index])
                        row[prefix+'_clipping_known']=bool(metadata['clipping_flags_known'][index])
                        row[prefix+'_feedback_clipped']=bool(metadata['teacher_feedback_clipped'][index].any())
                        row[prefix+'_native_clipped']=bool(metadata['teacher_native_clipped'][index].any())
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                stream.flush();os.fsync(stream.fileno());progress['proximity_queries_completed']+=1
        admit(args.request,args.clearance,args.clearance_sha256,source)
        report=dict(passed=True,evidence_diagnosis_completed=True,request_sha256=sha(args.request),
            rows=TOTAL,old_rows=OLD,new_rows=1018,alias_modes=summaries,normalization_algebra=algebra,
            proximity_queries=72,proximity_old_candidates=OLD,proximity_views_per_query=4,candidate_block=BLOCK,
            input_sha256={v['path']:v['sha256'] for v in subjects.values()},physical_arrays=request['physical_arrays'],
            source_sha256=request['source_sha256'],outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()},
            model_calls=0,native_steps=0,gradient_calls=0,optimizer_updates=0,training_selected=False,
            target_conflicts_repaired=False,rows_averaged_or_dropped=False,
            limitations=['Prelayer normalization aliases follow qualified source algebra; no ORT kernel or network output was executed.',
                'Signed-zero-equivalent groups are distinguished from bit-exact groups.',
                'Target spans are per-component ranges, not maximum pairwise target RMSE. Every member and target is retained.',
                'Proximity uses72 fixed first24-per-phase queries; absence of conflicts is not proof of global coverage.',
                'No new-state student prediction errors are available from this model-free diagnosis.'])
        save_json(output/'report.json',report)
    except BaseException as exc:
        save_json(output/'failure.json',dict(error=repr(exc),progress=progress,evidence_diagnosis_completed=False,automatic_retry=False))
        raise

if __name__=='__main__':main()
