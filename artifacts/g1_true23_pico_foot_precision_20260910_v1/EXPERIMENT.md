# PICO foot precision with preserved learner state

Prior goal turn: PROGRESS. Existing-PICO500 completes69.34/115.60s versus
50.60s untouched SONIC and30.48s at100, with substantially less root drift.
Leg/foot tracking still fails and walking accuracy regresses. All prior jobs
are terminal. Never promote a longer bad-tracking prefix or rerun fresh100.

New fixed-state encoder probe uses only actually recorded lower-body horizons.
Across2048 counterfactual pairs it finds no large reference change hidden by
token equality (largest invisible current-leg RMS0.006588rad; PICO0.001921rad).
This does not prove representation sufficiency, but does not justify replacing
the encoder or adding a precision bypass for the measured0.17rad leg error.

Hypothesis: the existing per-foot cost saturates and its main precision bonus
shares a denominator with root/upper error, weakening foot-placement learning
at observed10–25cm errors. Add one independent, nonnegative post-step bonus:
2/(1+existing_normalized_feet_cost/9), zero on any terminal/timeout. The scale
is15cm RMS, where the bonus is1; it is NOT an acceptance threshold. Preserve
every original reward, original29 target, source speed/frame, termination,
native dynamics/gain/limit, source encoder, seven-adapter actor and observation.
Do not impose a joint-position teacher or change physical target projection.

Use only unchanged saved walk002/003/PICO bank;008 remains optimizer-excluded,
previously seen development data. Import exact actor, critic, Adam moments and
counters from evaluated checkpoint047b7387f8268d2ff8032c8b74764de6a3f0321f4245814c0f5a909e805dc065.
The environment, episode/history state and RNG are initialized afresh and
explicitly are not an exact uninterrupted simulator resume. Original500 files
and source pins stay immutable. New checkpoint family records this new objective
and parent. No weights from failed trajectories are supervised teacher labels.

First two-update4env8step smoke from learner500, ending502. Check exact learner
import including nonempty Adam moments, independent reward arithmetic, all
trainable groups changing, frozen original weights and correct source timings.
Smoke is wiring only and is not the parent of the main run.

If that passes, one fixed500-additional-update128env16step run directly from
learner500, ending1000;1,024,000 new transitions. Save initial500,600 and1000.
Evaluate complete unchanged002/003/008/PICO requests at600 and1000 against500
and untouched SONIC. No warmup-frame cropping, hidden policy fallback, target
reanchoring, acceptance relaxation, new recordings or automatic extension.
Reject for deployment unless all full-motion/fidelity/physical gates pass;
timing, contact, source loss/recovery, standing entry/return and subsequent
supervised headset/hardware validation remain separate requirements.

No DDS, SSH, hardware controller, mode/arming, motor command or hardware export.
