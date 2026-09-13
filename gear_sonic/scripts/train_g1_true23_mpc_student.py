"""Bounded supervised student fit from complete physical teacher rollouts."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from gear_sonic.utils.g1_true23_mpc_student import make_actor,OFFSETS,KIND,sha256


def run(args):
    torch.set_num_threads(args.threads);torch.manual_seed(772);np.random.seed(772)
    args.output.mkdir(parents=True,exist_ok=False)
    report=json.loads((args.data/"report.json").read_text())
    allowed={c["case_id"] for c in report["case_summaries"] if c["expert_eligible"]}
    with np.load(args.data/"expert_samples.npz",allow_pickle=False) as a:
        X=a["features"].copy();ref=a["reference_joint_pos"].copy();target=a["target"].copy()
        case=a["case_id"].copy();frame=a["source_frame"].copy()
    assert set(np.unique(case))==allowed
    validation_case=max(allowed) if len(allowed)>1 else None
    train=np.ones(len(X),bool) if validation_case is None else case!=validation_case
    valid=~train
    mean=np.mean(X[train],axis=0).astype(np.float32);std=np.maximum(np.std(X[train],axis=0),.05).astype(np.float32)
    features=torch.from_numpy((X-mean)/std).to(args.device)
    labels=torch.from_numpy((target-ref)/.5).to(args.device)
    actor=make_actor().to(args.device)
    optimizer=torch.optim.AdamW(actor.parameters(),lr=3e-4,weight_decay=1e-5)
    ids=np.flatnonzero(train)
    phase=np.where(frame<261,0,np.where(frame<361,1,2))
    strata=[ids[phase[ids]==p] for p in np.unique(phase[ids])]
    request=dict(kind=KIND,device=args.device,steps=args.steps,batch_size=args.batch_size,train_samples=int(train.sum()),
                 validation_samples=int(valid.sum()),validation_case=validation_case,
                 split="whole physically executed rollout; same previously seen source recording, not recording generalization",
                 eligible_cases=sorted(allowed),excluded_failed_cases=report["total_cases"]-report["eligible_cases"],
                 architecture=[1023,256,256,23],activation="ELU",target="qref + .5*MLP(features), clipped only to native limits at inference",
                 goal_offsets=OFFSETS.tolist(),received_goal_frames=38,received_goal_buffer_seconds=.74,
                 no_clip_frame_time_inputs=True,phase_balancing="equal lifecycle phase sampling; phase metadata not an input",
                 input_hashes={str(p):sha256(p) for p in (args.data/"report.json",args.data/"expert_samples.npz",Path(__file__),Path(__file__).resolve().parents[1]/"utils/g1_true23_mpc_student.py")},
                 simulator_qualified=False,hardware_authorized=False)
    (args.output/"request.json").write_text(json.dumps(request,indent=2)+"\n")
    (args.output/"trainer_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (args.output/"student_snapshot.py").write_bytes((Path(__file__).resolve().parents[1]/"utils/g1_true23_mpc_student.py").read_bytes())
    best=float("inf");records=[];started=time.perf_counter()
    for step in range(1,args.steps+1):
        picked=np.concatenate([np.random.choice(part,args.batch_size//len(strata),replace=True) for part in strata])
        output=actor(features[picked]);loss=torch.mean((output-labels[picked])**2)
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),10.);optimizer.step()
        if step%100==0 or step==args.steps:
            with torch.inference_mode():
                predictions=actor(features)
                train_rmse=float(torch.sqrt(torch.mean((predictions[train]-labels[train])**2))*.5)
                val_rmse=float(torch.sqrt(torch.mean((predictions[valid]-labels[valid])**2))*.5) if valid.any() else None
            record=dict(step=step,train_target_rmse_rad=train_rmse,validation_target_rmse_rad=val_rmse,
                        elapsed_s=time.perf_counter()-started)
            records.append(record);print(json.dumps(record),flush=True)
            score=val_rmse if val_rmse is not None else train_rmse
            if score<best or step in (500,1000,args.steps):
                snapshot=dict(kind=KIND,goal_offsets=OFFSETS.tolist(),actor_state={k:v.detach().cpu() for k,v in actor.state_dict().items()},
                              feature_mean=torch.from_numpy(mean),feature_std=torch.from_numpy(std),step=step,request=request,
                              training_metric=record,full_body_tracking_qualified=False,hardware_authorized=False)
                if score<best:
                    best=score;torch.save(snapshot,args.output/"student_best.pt")
                if step in (500,1000,args.steps):torch.save(snapshot,args.output/f"student_{step:05d}.pt")
            (args.output/"metrics.json").write_text(json.dumps(records,indent=2)+"\n")
    (args.output/"outcome.json").write_text(json.dumps(dict(completed=True,steps=args.steps,best_label_rmse_rad=best,
                                                          closed_loop_validation_pending=True,full_body_tracking_qualified=False,
                                                          elapsed_s=time.perf_counter()-started),indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=2000);p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--threads",type=int,default=2);p.add_argument("--device",default="cpu")
    run(p.parse_args())
