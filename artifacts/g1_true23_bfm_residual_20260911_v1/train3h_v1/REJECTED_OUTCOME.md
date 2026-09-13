Residual v1 rejected after full CPU replay of checkpoints200/400/600. Training stopped after the complete checkpoint800 was saved and checked. No real hardware tested or authorized.

All three milestones fail full-body tracking. walk008 completes469/1114 controls at200, then415/1114 at400 and600; failures are measured physical joint-range violations. PICO completes its duration but still exceeds actual joint range (.00537/.00347/.00492rad). At600, full-source root p95 remains .417m on walk002, .297m on PICO and .609m on walk003; original29 hand/head and native foot errors remain substantial. These outcomes do not support deploying this residual policy.

Metrics are mixed, not uniformly worsening: PICO legRMSE at200/400/600 is .1884/.1976/.1845rad. Low training reward/loss trends or selected improved joints cannot replace full CPU tracking evidence. walk008 is a previously seen development evaluation-only recording; walk002/walk003/PICO are training recordings.

Complete checkpoint: residual_00800.pt (4307031 bytes; SHA256 3b14bf05df7662569119d39ffa46811e0b78f06846abf3243b800578159af707). CPU torch.load(weights_only=True) passed, completed_updates=800. Last metric update=800. Separate parent watcher owns full CPU800 evaluation; no800 tracking result is claimed here.

Authorized SIGINT sent to the exact trainer PID363 at 2026-09-10T15:09:05.319034+00:00 after checkpoint ZIP verification and a stable-size wait. PID absent by 2026-09-10T15:09:07.322866+00:00. Trainer session68204 closed with expected KeyboardInterrupt exit1; stop watcher70280 closed exit0. Existing handler saved residual_interrupted.pt, wrote outcome.json and closed the environment. Interrupted snapshot contains partial next-rollout/normalizer state; residual_00800.pt is the complete milestone.

All original checkpoints, metrics, request and outcome remain unchanged. experiment_rejection.json records the stop receipt, last metric, portable-load result and exact full CPU200/400/600 summaries with source hashes. No unchanged extension is scheduled. MPC distillation remains a plan until physically executed teacher results exist.
