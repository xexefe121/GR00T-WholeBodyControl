"""Fit a bounded BFM residual using only accepted physical teacher labels."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from gear_sonic.utils.g1_true23_bfm_mpc_student import make_actor, KIND, FEATURES, BASE_CONTRACT
from gear_sonic.utils.g1_true23_mpc_student import OFFSETS, sha256


def run(args):
    torch.set_num_threads(1); torch.manual_seed(773); np.random.seed(773)
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.data / "report.json").read_text())
    assert source["kind"] == KIND and source["base_contract"] == BASE_CONTRACT
    assert sha256(args.data / "labels.npz") == source["labels_sha256"]
    with np.load(args.data / "labels.npz", allow_pickle=False) as archive:
        X, Y, case, frame = [archive[key].copy() for key in ("features", "residual_target", "case_id", "source_frame")]
    allowed = source["accepted_case_ids"]
    assert set(np.unique(case)) == set(allowed) and len(allowed) >= 2
    validation_case = max(allowed); train = case != validation_case; valid = ~train
    mean = X[train].mean(0).astype(np.float32)
    std = np.maximum(X[train].std(0), .05).astype(np.float32)
    features = torch.from_numpy((X - mean) / std).to(args.device)
    labels = torch.from_numpy(Y.astype(np.float32)).to(args.device)
    actor = make_actor().to(args.device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=3e-4, weight_decay=1e-5)
    ids = np.flatnonzero(train)
    phase = np.where(frame < 261, 0, np.where(frame < 361, 1, 2))
    strata = [ids[phase[ids] == p] for p in np.unique(phase[ids])]
    request = dict(kind=KIND, base_contract=BASE_CONTRACT, radius_rad=args.radius, steps=args.steps,
                   device=args.device, cpu_threads=1, architecture=[FEATURES, 256, 256, 23],
                   target="radius*tanh(MLP) added to frozen BFM unclipped target",
                   labels="expert preclip command minus frozen BFM on same observed state and expert combined history",
                   accepted_case_ids=allowed, excluded_failed_cases=source["rejected_cases"],
                   train_samples=int(train.sum()), validation_samples=int(valid.sum()), validation_case=validation_case,
                   split="held-out initial perturbation of same previously seen recording; not recording generalization",
                   phase_balancing="standing/acquisition/source equally sampled; phase not a network input",
                   goal_offsets=OFFSETS.tolist(), declared_received_goal_buffer_seconds=.74,
                   no_clip_frame_or_time_inputs=True,
                   fraction_labels_outside_radius=float(np.mean(np.abs(Y) > args.radius)),
                   bounded_oracle_target_rmse_rad=float(np.sqrt(np.mean((np.clip(Y, -args.radius, args.radius) - Y) ** 2))),
                   provenance={str(path): sha256(path) for path in (args.data / "report.json", args.data / "labels.npz",
                                                                   Path(__file__), Path(__file__).resolve().parents[1] / "utils/g1_true23_bfm_mpc_student.py")},
                   full_body_tracking_qualified=False, hardware_authorized=False)
    (args.output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    started = time.perf_counter(); records = []; best = float("inf")

    def checkpoint(step, record):
        return dict(kind=KIND, base_contract=BASE_CONTRACT, radius_rad=args.radius,
                    goal_offsets=OFFSETS.tolist(), actor_state={k: v.detach().cpu() for k, v in actor.state_dict().items()},
                    feature_mean=torch.from_numpy(mean), feature_std=torch.from_numpy(std),
                    step=step, request=request, metric=record, full_body_tracking_qualified=False, hardware_authorized=False)

    torch.save(checkpoint(0, {}), args.output / "residual_00000.pt")
    for step in range(1, args.steps + 1):
        picked = np.concatenate([np.random.choice(part, args.batch_size // len(strata), replace=True) for part in strata])
        prediction = args.radius * torch.tanh(actor(features[picked]))
        loss = torch.mean((prediction - labels[picked]) ** 2) / args.radius ** 2
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.); optimizer.step()
        if step % 100 == 0 or step == args.steps:
            with torch.inference_mode():
                prediction = args.radius * torch.tanh(actor(features))
                train_rmse = float(torch.sqrt(torch.mean((prediction[train] - labels[train]) ** 2)))
                val_rmse = float(torch.sqrt(torch.mean((prediction[valid] - labels[valid]) ** 2)))
            record = dict(step=step, train_target_rmse_rad=train_rmse, validation_target_rmse_rad=val_rmse,
                          elapsed_s=time.perf_counter() - started)
            records.append(record); print(json.dumps(record), flush=True)
            saved = checkpoint(step, record)
            if val_rmse < best:
                best = val_rmse; torch.save(saved, args.output / "residual_best.pt")
            if step in (500, 1000, args.steps):
                torch.save(saved, args.output / f"residual_{step:05d}.pt")
            (args.output / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    for name, path in (("trainer_snapshot.py", Path(__file__)),
                       ("student_snapshot.py", Path(__file__).resolve().parents[1] / "utils/g1_true23_bfm_mpc_student.py")):
        (args.output / name).write_bytes(path.read_bytes())
    (args.output / "outcome.json").write_text(json.dumps(dict(steps=args.steps, best_label_rmse_rad=best,
        elapsed_s=time.perf_counter() - started, closed_loop_validation_pending=True,
        full_body_tracking_qualified=False, hardware_authorized=False), indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1000); p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--radius", type=float, default=.25); p.add_argument("--device", default="cpu")
    run(p.parse_args())
