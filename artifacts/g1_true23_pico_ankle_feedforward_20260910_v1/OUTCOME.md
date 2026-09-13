# Bounded ankle correction implemented; full-body controller REJECTED

Full fixed trial29117 EXIT0 (successful evidence capture, failed controller).
No job remains active. Native model/gains/targets/total effort limits and all
recorded sources stay unchanged; original20ms range check is retained. The
100ms target-only experiment is not adopted. No policy training or hardware path.

One verified-2.5Nm right-ankle-roll correction at PICO control1880 repairs the
original30.60s local range stop. Original behavior is exact up to this point.
The controller then runs123 further controls without feedforward and loses
balance at33.08/115.60s. At that stop, actual pelvis height is.109046m versus
target.734578m; actual tilt1.533699rad versus target.234855rad. Root position
error grows to2.072541m. This is not a commanded low pose or another range
filter rejection: the existing absolute-height diagnostic detects a fall.

The single correction preserves the local ankle bound, not future balance.
It is not evidence that feedforward alone caused the fall or that a stronger
correction would solve it. No automatic offset sweep follows this rejection.

| Clip | Completed/requested controls | Source seconds | Leg RMSErad | Root p95m | Relative foot p95mL/R |
| --- | --- | --- | --- | --- | --- |
| PICO |2004/6530|33.08/115.60|.186556|.397583|.178344/.174553|
|002|1417/1417|13.34/13.34|.218125|.774332|.223886/.184748|
|003|1104/1569|15.08/16.38|.221970|1.629224|.210221/.187084|
|008|681/1114|6.62/7.28|.226228|2.282379|.225430/.346740|

All4 full-body tracking screens FAIL.003 cannot find a verified bounded torque
correction;008 gains1.82source seconds after one correction, then rejects too.
At the exact old30.60s PICO prefix, every reported tracking metric is unchanged.
No later, worse full-motion result is hidden by that equality.

Independent27238 EXIT0:

-52,060 actual2ms steps and5,206 landmark vectors reproduced exactly;
 384 independent policy re-inferences exact.
-All52,060 total PD+feedforward arithmetic/effort clips and5,206 accepted range
 predictions checked;386 fresh predictions/3,860 substeps exact, including
 both actual feedforward decisions. Native actual range excess0, maximum
 velocity ratio.821186, engine total effort ratios<=1.
-All4,991 pre-intervention/original-prefix controls match earlier policy,
 observations, states and torques. Other cloned benchmark helper ASTs unchanged.
-All measured filter calls under20ms; maximum16.999ms. This excludes inference,
 scheduling, transport and hardware, so it is NOT50Hz or live qualification.

38 focused tests77037 EXIT0/12.87s; scoped Ruff E/F pass. Final PICO report
1e3d84212233ac23e54507efff92b9231e6d278100302c8e6f42f78d384b6792;
trace c821a9284ab5b8b85a4cc4a2ae104999048189a1630e5e39761ce0391d116e3d.
Exact balance/target samples:eval1000_v1/pico_balance_failure.json.

## Decision and next bounded work

Do not promote either new guard/controller. The diagnosed ankle authority issue
is real, but fixing that local stop does not fix learned native23 balance or
feet. More ankle-horizon/correction sweeps are not justified by these results.

Next audit the existing learner's actual state/reward distribution at the
pre-failure PICO phase, including lower-joint posture versus original29 torso/
hand intent and native foot placement. Check earlier reset/full-decoder/waist-
compensation experiments before proposing a new training or retarget branch.
Do not assert unseen training states or a reward conflict without measuring it.
This is a new decision boundary, not authorization for another unchanged run.

Goal ACTIVE. Full-body live VR NOT READY. No original implementation edits,
physical robot/DDS/mode command, safeguard bypass, commit, push or export.
