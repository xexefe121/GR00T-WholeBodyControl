# Fixed anticipatory-ankle hypothesis — simulation only

The prior solved-force diagnostic identifies insufficient ankle-only target
authority at final1000's30.60s rejection and a20ms range checker that has no
braking/recursive-feasibility guarantee. Earlier100ms saved-state probes on
the different momentum policy also warned before its20ms stop. No longer-
horizon closed-loop controller has yet been accepted. The earlier coordinated
leg search is rejected for500ms computation and>2rad target changes.

One fixed change: retain every native joint's10-substep/20ms reserve checks,
and also require all four ankle axes to satisfy that same reserve throughout
50 substeps/100ms under each constant candidate PD target. Use independent
zero-warmstart state copies. There are no future source/policy samples. This
extra conservative check is not a true prediction of future policy decisions
or proof of recursive viability. No fallback after failure.

Use the same bounded inward search, maximum16 unique predictions. Correct
its numerical endpoint direction so an already stronger nominal target is
never weakened. Preserve all untouched raw action bytes and skip duplicate
decoded targets; when no inward authority remains, fail immediately. These
changes do not expand target/effort limits. No coordinated-joint optimizer,
gains, physical parameters, source timing, retargeting or learned weights change.

First test fixed saved final1000 controls1830..1880 (last1s plus rejected state)
without applying candidates to the saved trajectory. Then request the complete
unchanged PICO/walk002/walk003/walk008 lifecycles with final1000 checkpoint
ac2f844894e3e53c364ae466966a64db556ee255c6ab72e736fe15b87adb19fe.
Start each complete trial at the original initial state only; never resume
partway through a physical rollout. Preserve every failure and compare equal
source prefixes against final1000's20ms baseline and parent500. No crop/retime.

Qualification still requires full source/standing return, original leg/foot/
hand/head fidelity, physical range/effort/velocity, actual-clock timing and
pause/disconnect/recovery. Do not promote a longer prefix as readiness. Record
filter timing/interventions/target changes. If a fixed hypothesis fails, no
automatic horizon sweep or new training run. Keep originals and failed evidence
immutable. No hardware/DDS/SSH/mode/arming/motor commands or exports.
