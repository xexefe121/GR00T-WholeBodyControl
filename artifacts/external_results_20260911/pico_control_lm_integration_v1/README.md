# Optional third control-LM restoration

Shared source changes: core accepts a backward override only for non-hard private optimization; the hard main solver rejects it. New restoration_lm helper delegates original guided/K0 attempts unchanged, then optionally tries one control-LM K0 solve from first guided ordinary-final targets with original shifted targets as the regularization anchor. Strict native/nominal certification remains mandatory before ordinary MPC resumes.

Evaluator --restoration-control-lm-retry requires --restoration plus its existing hard-feasibility/H30 prerequisites. Disabled option keeps original helper import/call and request/result fields. Enabled option records complete third-attempt policy, source snapshot, all generated merit/target sequences, proposal state/history, and attempt count. Main tracking objective, source timing, native limits and strict execution guards remain unchanged.

Validation: 53 focused native/default tests passed, zero skips. Final 10 third-LM tests passed after test-only formatting. Ruff all four changed/new source/test files passed. WSL runtime does not contain Ruff; Windows Python Ruff was used. Native tests used pinned MuJoCo3.2.3 and original mjbatch runtime, BLAS1.

Frozen repo: thirteen import files, ten direct shared source files hash bound in frozen_source_hashes.json. integration.diff compares against old full canonical PICO source snapshots. New helper is behaviorally/AST identical to the independently reviewed bounded helper (module docstring changed); core is byte-identical to its cleared bounded extension.

Bounded evidence remains limited: actual3800 to3900 completes all1000 physics steps with root replay bitexact; root original-intent checks fail yaw/feet/leg tracking. This motivates one full canonical experiment, not a recovery/full-body qualification claim.

Full run prepared separately as pico_full_control_lm_v1, canonical new native MjData at declared v4 frame10, all6530 lifecycle/5780 source controls. No old trace or private proposal is injected. Same old full PICO H30/5/commit5/eightthreads/.1 feedback/allmargin.05/2000/relativefeet400/fresh and recorded original BFM seeds; only controller change is enabled third fallback. Frozen absolute script import, durable hidden Windows wrapper, .NET SHA256 input checks, separate stdout/stderr, PID/start/exit receipts. Review before the single launch.
