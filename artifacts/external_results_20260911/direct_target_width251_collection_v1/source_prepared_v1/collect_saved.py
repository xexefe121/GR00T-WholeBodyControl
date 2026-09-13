"""Conditional saved-file collector; no inference, native, or fitting imports."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from direct_features import DirectFeatures
from input_schema import archive_headers,load_numeric,MOTION,ORIGINAL
from collection_adapter import collect
from collection_math import difference_function
from qualification_gate import admit,local,read,sha

def load(path):
    archive_headers(path)
    with np.load(path,allow_pickle=False) as archive:return {key:archive[key].copy() for key in archive.files}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);args=parser.parse_args()
    request=read(args.request);admit(request,Path(__file__).resolve().parent)
    if np.__version__!='1.26.4':raise RuntimeError('Qualified NumPy1.26.4 required for actual target-map byte parity')
    output=local(request['output']);output.mkdir(exist_ok=False)
    try:
        subjects=request['subjects'];paths={k:local(v['path']) for k,v in subjects.items()}
        motion,_=load_numeric(paths['motion'],MOTION);original,_=load_numeric(paths['original29'],ORIGINAL)
        contract=read(paths['contract'])
        if motion['fps'][0]!=50 or original['fps'][0]!=50:raise ValueError('Original50Hz goal source')
        if len(contract['body_names'])!=24 or len(contract['joint_names'])!=23:raise ValueError('Original native23 contract')
        for quats in (motion['body_quat_w'],original['source_task_quaternion_wxyz']):
            if np.max(np.abs(np.linalg.norm(quats,axis=-1)-1))>1e-6:raise ValueError('Reference quaternion norm')
        plans={entry['control']:load(local(entry['path'])) for entry in request['plans']}
        rows=collect(load(paths['main_trace']),load(paths['hold_trace']),load(paths['boundary_snapshot']),contract,plans,
                     read(paths['plan_records']),DirectFeatures(motion,original,contract),difference_function(paths['core']))
        np.savez_compressed(output/'expert_rows.npz',**rows)
        shutil.copyfile(paths['normalization'],output/'normalization.npz')
        if sha(output/'normalization.npz')!=subjects['normalization']['sha256']:raise ValueError('Frozen normalization copy')
        # No changed producer or receipt may be accepted after the saved calculations.
        admit(request,Path(__file__).resolve().parent)
        summary={str(phase):dict(rows=int(np.sum(rows['phase']==phase)),
            feedback_clipped_rows=int(np.sum(np.any(rows['feedback_clipped'][rows['phase']==phase],axis=1))),
            native_clipped_rows=int(np.sum(np.any(rows['native_clipped'][rows['phase']==phase],axis=1)))) for phase in range(3)}
        report=dict(passed=True,collection_completed=True,model_fitting_authorized=False,rows=1018,control_start=251,
            control_stop_exclusive=1269,fresh_student_state_queries=1,connected_expert_rows_after_first=1017,
            source_frames='global_control+11',current_features=1000,context_features=323,features=1323,
            target_semantics='actual applied expert native joint target, float64 radians',
            map_semantics='saved committed expert feedback map, not independent fresh replan at every endpoint',
            request_sha256=sha(args.request),input_sha256={v['path']:v['sha256'] for v in subjects.values()},
            plan_subjects=request['plans'],source_sha256=request['source_sha256'],phase_summary=summary,
            outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()},
            task_model_calls=0,bfm_actor_calls=0,bfm_backward_calls=0,native_steps=0,replans=0,optimizer_updates=0)
        with (output/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    except BaseException as exc:
        with (output/'failure.json').open('x') as f:json.dump(dict(type=type(exc).__name__,message=str(exc),collection_completed=False),f,indent=2)
        raise

if __name__=='__main__':main()
