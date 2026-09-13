# Anticipatory target-only controller REJECTED

Fixed50-substep/100ms ankle horizon tested; no sweep or training extension.
Original every-joint20ms checks,0.0019rad reserve, gains, bounds, checkpoint and
all recorded inputs stay unchanged. New directional endpoint handling avoids
weakening an already stronger nominal target and skips duplicate predictions.
The original executed range-preview source remains unchanged.

Saved-state matrix83730:51 copied final1000 states,50 accepted/11 interventions;
old final rejected state still has no inward target authority. Full trial83730
EXIT0 (evidence writer succeeded; controller NOT qualified):

| Clip | Completed/requested controls | Source seconds | Leg RMSErad | Root p95m | Relative foot p95mL/R |
| --- | --- | --- | --- | --- | --- |
| PICO |1945/6530|31.90/115.60|.163440|.247471|.142856/.157630|
|002|1417/1417|13.34/13.34|.214213|.944246|.219474/.179697|
|003|1569/1569|16.38/16.38|.225312|1.785582|.210013/.188355|
|008|667/1114|6.34/7.28|.216846|2.274129|.214108/.259122|

All4 full-body tracking screens FAIL. PICO gains1.30s over final1000's20ms
baseline and003 completes its lifecycle, but neither is successful teleop.
On the identical30.60s PICO prefix, original20ms→100ms root p95 worsens
.190386→.243868m; leg RMSE.160060→.160091rad, arm.269828→.271432rad.
Relative feet improve slightly.127001/.145065→.124378/.142716m; still fail.
New PICO rejection's nominal20ms prediction is safe, while its100ms constant-
target projection is not. This conservative stop is NOT proof that the next
actual policy action would inevitably violate a limit.

Independent audit96360 EXIT0:55,980 actual2ms steps,5,598 landmark vectors,
384 policy re-inferences exact. All5,598 saved accepted prediction constraints
recomputed;384 fresh predictions/19,200 substeps exact. Actual physical joint
range excess0; maximum velocity ratio.798422; total motor effort ratios<=1.
Evidence consistency does not qualify failed tracking or incomplete motions.

Filter alone exceeds20ms on252 controls:156PICO,23/13/60 walking. Maximum
63.65/64.47/71.85/76.82ms respectively. These unpaced measurements are not an
actual-clock test and already rule out live promotion of this implementation.
27 focused tests passed; subsequent combined38-test suite also passed. No
physical transport, model/limit/gain change, checkpoint promotion or export.

Next distinct hypothesis measures available native ankle actuator torque,
not a longer target-horizon sweep. Fixed copied-state probes are separate in
g1_true23_pico_ankle_torque_authority_20260910_v1. Goal ACTIVE; deployment NOT READY.
