# Local actuator authority exists; not a deployment result

Fixed8-probe matrix18097 EXIT0. No actual trajectory or hardware executed.
Final1000's zero-offset first10states/velocities/torques reproduce the previous
rejected nominal probe exactly. All targets, gains and total effort limits
remain unchanged. Complete velocity/total-torque audit also passes.

At final1000's1880-control state, right-ankle-roll feedforward-2.5Nm removes
the original0.003895rad reserve excursion over20ms. Position after20ms becomes
.249623rad with inward velocity-.793844rad/s; zero-offset gave.263795rad with
outward+.142889rad/s. The tested100ms constant-target trajectory also remains
inside all joint reserves. Total ankle torque is-18.41 to-14.87Nm over20ms;
maximum native velocity ratio.281955. Original35Nm total effort bound is intact.
The-5/-10Nm probes also pass these local checks; they are not chosen as gains.

Parent500's later3817-control state is different: every0/-2.5/-5/-10Nm probe
still violates the ankle reserve. Its nominal position target points outward,
and no change of that target was part of this fixed diagnostic. Do not infer
that the parent state is globally unrecoverable or repeat an undeclared sweep.

Results justify testing a separately versioned bounded-torque controller path,
not claiming generic recovery. That full-lifecycle branch is in
g1_true23_pico_ankle_feedforward_20260910_v1 and was subsequently REJECTED:
it fixes the local ankle failure but loses whole-body balance. No hardware
transport, gain/limit change, automatic training or checkpoint export.
