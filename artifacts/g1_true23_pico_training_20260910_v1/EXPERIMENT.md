# Existing PICO data coverage: bounded SONIC adaptation

Previous normal-core training explicitly excluded PICO-FreeDancing and used
walking002/003 plus DadDance. Its100-update result worsened PICO tracking.
Selected native124 also fails all eight full PICO/walking comparison requests;
do not use it as teacher or replace SONIC with it.

Prepare a new versioned research bank using only already saved walk002,
walk003 and PICO-FreeDancing original29/native23 lifecycles. Preserve every
original-speed sample and all29-axis hand/head intent. Exclude walk008 from
optimizer data; it is a previously inspected development check, not an untouched
test. PICO is explicitly training data now, not a held-out generalization claim.
Retain the exact standing/source/planned-endpoint-return timelines, task scoring,
native model, gains, limits, source codec, zero absent-joint measurements,
normal47-sample920ms horizon, frozen SONIC base and all seven rank16 adapters.

First run a two-update four-environment wiring smoke on this new bank. Verify
received-only references, original29 geometry, true23 physics, gradients and
frozen base. No learned-policy claim from this smoke and do not resume its actor.

Then one fresh500-update128-environment16-control PPO run:1,024,000 transitions.
Save0/100/250/500 milestones without reinitializing actor, critic, optimizer or
simulation between them.100 is the matched-budget data-coverage comparison;
500 provides a fixed, larger exposure budget instead of more independent short
restarts. No automatic extension or checkpoint cherry-picking. Same seed,
optimizer, rewards, termination and precision as the verified normal-core run.
Stop on invalid physics/numerics; none of the physical gates may be relaxed.

The new launcher must truthfully record PICO training ownership. Bound expensive
input telemetry to explicit sampled control indices; retain aggregate counts,
all gradient/reward audit data and complete checkpoint identities. It must not
retain unlimited full obs930 arrays for every simulated environment step.

Evaluate complete fixed lifecycles at100 and500, including all115.60s of PICO
and walk008, against the unchanged SONIC baseline. No source cropping/retiming,
no substitution of survival or prefix scores for full tracking. Two consecutive
milestones worsening both PICO and walking mean reject this branch, no further
updates. Any improvement remains SIM-only until full timing, contact/limits,
entry/return and fault/recovery checks pass. Never run hardware here.
