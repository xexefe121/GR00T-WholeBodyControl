# Existing-source, actual-clock localhost SIM experiment

This experiment addresses a known runtime defect: the earlier live consumer
waited up to500ms for a packet with no physics integration, then omitted its
balance tail from the optional trace. The prior virtual-clock experiment fixes
that conceptual boundary but does not test wall-clock execution or transport.

New additive implementation: paced_sim_runtime.py, run_g1_true23_paced_sim.py,
and test_g1_true23_paced_saved_stream.py. The same paired breadth25 controller,
native23 model, gains, effort/range limits and complete saved walk002 source stay
unchanged. First two packets initialize the source pose; a40ms actual-clock
buffer absorbs ordinary arrival jitter without retiming source motion. No
standing acquisition, SONIC reacquisition or firmware handback is claimed.

An acknowledged localhost XPUB/SUB connection sends all original packets at50Hz
in the same monotonic clock domain. Every command records real start/finish,
deadline, source timestamp/index and every unchanged2ms physics substep. The ten
substeps are batched within a wall-clock20ms control, not individually paced.
No hardware channel exists. This is not an OS hard-real-time guarantee.

Trials: complete source plus250 balance controls;500ms source pause at200;
omitted packet200; malformed JSON at200. A fault must latch balance without
implicit SONIC re-entry. Fault cases intentionally stop source tracking. The
consumer never waits for another packet to decide its next simulation tick.

Any whole-period late wake or finish past the next deadline is a TIMING FAILURE.
Balance uses the existing approved timeout activation; runtime_fault separately
records the scheduler cause. On overrun the schedule rebases, and no catch-up
burst or timestep change erases lost wall time. No failed trial is overwritten.

Screens: exact requested physics counts/times, actual joint/effort/velocity
limits, height>=.45m, tilt<=1rad, timely fault classification, strict latch,
complete sent-message hash sequence received without loss, and no compute
deadline failures. Source+EOF qpos/qvel/time must match the earlier virtual case
exactly to show unchanged policy behavior. This equality also means the known
failed leg/foot/arm tracking remains failed. Transport screens are not tracking
screens and cannot make the overall goal or deployment ready.

Read-only checks before this experiment ruled out two proposed explanations:
original SONIC-transfer uses an analytic (not learned) codec; existing lower240
perturbation evidence already shows nonconstant encoder leg tokens. Fresh
normal64 return/critic computation gives explained variance.91704 at update100;
large raw value loss alone does not establish a broken critic. No new policy
training, architecture change or tracking improvement is claimed this turn.
