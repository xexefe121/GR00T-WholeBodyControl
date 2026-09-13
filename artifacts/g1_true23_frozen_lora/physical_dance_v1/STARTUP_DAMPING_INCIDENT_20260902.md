# G1 true23 startup-damping incident

Status: physical qualification invalidated; no automatic retry authorized.

The 2026-09-01 direct-dance controller released Unitree motion mode before its
ten-frame policy/history warmup completed. `GantrySafetyCore::BuildCommand`
returned `BuildDampingCommand()` while unarmed, so the writer emitted commands
with `kp=0`, `kd=1.5`, and `tau=0`. G1 consequently entered the observed
dumped/limp mode as soon as the session started.

The decisive timing in
`live_dance.direct.verify.execution.b0180a162573.jsonl` is:

- `motion_mode_released`: monotonic 352535644115 ns
- `lowcmd_publisher_created`: monotonic 352639899476 ns
- `first_policy_ready_for_arm`: monotonic 375392119291 ns
- release-to-policy-ready gap: 22.856475176 seconds

The terminal PASS in that record counted later policy commands and terminal
damping frames. It did not classify pre-arm packets, so it was a false
readiness result. Both physical execution records from that session are
superseded and must not authorize another run.

Required replacement invariant:

1. Prewarm policy/history while Unitree motion mode remains active.
2. Permit zero LowCmd writes before release.
3. Sample current posture before release.
4. Make first post-release packet a positive-`kp`, zero-feedforward hold.
5. Require at least 25 successful hold packets and zero startup damping packets
   before enabling either direct `DANCE` or wireless L2/A arming.
6. Reserve `kp=0` damping for stop/fault only.

New physical qualification requires a fresh explicit `DANCE` command and new
hold-first execution evidence. This incident report does not authorize motion.
