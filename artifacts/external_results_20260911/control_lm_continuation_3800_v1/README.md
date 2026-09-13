# Actual3800 bounded control-LM continuation

ONE authorized continuous100-control2s simulation run completed; original full PICO remains incomplete. No full-source/recovery/deployment qualification.

- Native MuJoCo3.2.3, original23-DOF model/limits and500Hz PD; H30/5 main iterations/commit5/eight Batch threads/.1rad feedback clip, all-margin.05/2000 and same-world relative feet400.
- Single initialization from canonical PICO actual3800 full integration/history. No later state rewrites. Source69 to71s, actual3800 to3900,1000 physics steps; all imminent and actual substep checks pass. No warning/clock failure.
- At3800, unchanged guided and original-warm K0 fail and reproduce archived proposals bitexact. Third control-LM ordinaryfinal starts first guided final, original-warm regularization anchor, private K0 rollouts. It reproduces the independently certified witness bitexact, then main MPC optimizes and executes normally. Saved witness never injected.
- At3805 guided restoration succeeds. No later restoration through3900. Total two triggers, one K0 retry and one third-LM retry.
- Producer elapsed84.324s for2s simulated: offline. Report completed=true/failure=null; recovery_qualified=false/full_source_replay=false/deployment_ready=false.
- All14 frozen source hashes and10 original shared source hashes remain unchanged. Saved evidence shape/clock/source binding checks pass; root independent replay and original-intent metrics pending.

Durable hidden process wrapper in sibling control_lm_continuation_3800_v1_process: PID23388, started2026-09-11T03:01:46Z, exited0 at03:03:35Z. Exact command/source/input hashes, separate stdout/stderr, PID/start/exit receipts retained. Wrapper prevents duplicate output/log use.

Trace SHA256: 0ce95ddf66876efa78a00b3c7148004e95c52d8989d9693656a13be218a6525a
Request SHA256: 4052e115ace4a478bfd714bf5eb3e1c781b6cbc850dc0765f8d349969b97c62e
