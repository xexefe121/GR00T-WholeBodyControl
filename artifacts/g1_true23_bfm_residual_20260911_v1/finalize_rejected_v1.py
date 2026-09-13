"""Preserve a separate, portable record of the authorized residual-v1 stop."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch


root = Path(__file__).resolve().parents[2]
experiment = root / "artifacts/g1_true23_bfm_residual_20260911_v1"
run = experiment / "train3h_v1"
checkpoint = run / "residual_00800.pt"
state = torch.load(checkpoint, map_location="cpu", weights_only=True)
print("checkpoint keys", sorted(state))
assert state["completed_updates"] == 800
receipt = json.loads((experiment / "authorized_stop_after800_receipt.json").read_text())
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == receipt["checkpoint_sha256"]
assert not Path("/proc/363").exists(), "Original trainer PID unexpectedly exists"
outcome = json.loads((run / "outcome.json").read_text())
assert outcome["updates"] == 800
assert outcome["error"] == "KeyboardInterrupt()"
metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
assert metrics[-1]["update"] == 800
milestones = []
for update in (200, 400, 600):
    path = root / f"artifacts/teleop_six_hour_20260910/residual_milestones_v1/residual_{update:05d}/summary.json"
    summary = json.loads(path.read_text())
    milestones.append({"update": update, "source": str(path.relative_to(root)),
                       "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "full_body_tracking_qualified": summary["full_body_tracking_qualified"],
                       "cases": summary["cases"]})

record = {
    "status": "rejected_after_full_cpu_milestones",
    "written_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "Checkpoints200/400/600 do not qualify full-body tracking; eval-only walk008 repeatedly fails actual physical joint limits, PICO still exceeds actual joint range, and root/feet/original29 hand-head tracking remains poor.",
    "original_run_outcome_preserved": True,
    "authorization": "Parent requested stop only this residual run after the next complete checkpoint800; no restart authorized.",
    "last_complete_checkpoint": str(checkpoint.relative_to(root)),
    "checkpoint_sha256": receipt["checkpoint_sha256"],
    "checkpoint_bytes": checkpoint.stat().st_size,
    "portable_weights_only_cpu_load_passed": True,
    "completed_updates": state["completed_updates"],
    "last_training_metric": metrics[-1],
    "stop_receipt": receipt,
    "trainer_pid_absent": True,
    "trainer_exec_session": 68204,
    "trainer_session_closed": True,
    "trainer_exit_code": 1,
    "trainer_exit_explanation": "Expected KeyboardInterrupt after SIGINT; existing exception/finally saved interrupted state, wrote outcome, and closed environment.",
    "stop_watcher_exec_session": 70280,
    "stop_watcher_session_closed": True,
    "stop_watcher_exit_code": 0,
    "interrupted_snapshot_note": "Saved during the next rollout after completed update800. Use residual_00800.pt as the complete milestone; interrupted snapshot includes partial subsequent rollout/normalizer state.",
    "evaluation_update800": "Separate parent CPU watcher will evaluate complete checkpoint800; no800 tracking result inferred here.",
    "split_note": "walk002/walk003/PICO are training recordings; walk008 is a previously seen development evaluation-only recording, not an untouched test set.",
    "trend_note": "Metrics are mixed rather than monotonically worsening. PICO legRMSE at200/400/600 is .1884/.1976/.1845rad; full physical and intent requirements remain unmet.",
    "simulator_qualified": False,
    "hardware_authorized": False,
    "milestones": milestones,
}
(run / "experiment_rejection.json").write_text(json.dumps(record, indent=2) + "\n")

lines = [
    "Residual v1 rejected after full CPU replay of checkpoints200/400/600. Training stopped after the complete checkpoint800 was saved and checked. No real hardware tested or authorized.",
    "",
    "All three milestones fail full-body tracking. walk008 completes469/1114 controls at200, then415/1114 at400 and600; failures are measured physical joint-range violations. PICO completes its duration but still exceeds actual joint range (.00537/.00347/.00492rad). At600, full-source root p95 remains .417m on walk002, .297m on PICO and .609m on walk003; original29 hand/head and native foot errors remain substantial. These outcomes do not support deploying this residual policy.",
    "",
    "Metrics are mixed, not uniformly worsening: PICO legRMSE at200/400/600 is .1884/.1976/.1845rad. Low training reward/loss trends or selected improved joints cannot replace full CPU tracking evidence. walk008 is a previously seen development evaluation-only recording; walk002/walk003/PICO are training recordings.",
    "",
    f"Complete checkpoint: residual_00800.pt ({checkpoint.stat().st_size} bytes; SHA256 {receipt['checkpoint_sha256']}). CPU torch.load(weights_only=True) passed, completed_updates=800. Last metric update=800. Separate parent watcher owns full CPU800 evaluation; no800 tracking result is claimed here.",
    "",
    f"Authorized SIGINT sent to the exact trainer PID363 at {receipt['signal_sent_utc']} after checkpoint ZIP verification and a stable-size wait. PID absent by {receipt['checked_utc']}. Trainer session68204 closed with expected KeyboardInterrupt exit1; stop watcher70280 closed exit0. Existing handler saved residual_interrupted.pt, wrote outcome.json and closed the environment. Interrupted snapshot contains partial next-rollout/normalizer state; residual_00800.pt is the complete milestone.",
    "",
    "All original checkpoints, metrics, request and outcome remain unchanged. experiment_rejection.json records the stop receipt, last metric, portable-load result and exact full CPU200/400/600 summaries with source hashes. No unchanged extension is scheduled. MPC distillation remains a plan until physically executed teacher results exist.",
]
(run / "REJECTED_OUTCOME.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"status": record["status"], "portable_load": True, "completed_updates": 800,
                  "checkpoint_sha256": receipt["checkpoint_sha256"],
                  "rejection_record": str(run / "experiment_rejection.json")}, indent=2))
