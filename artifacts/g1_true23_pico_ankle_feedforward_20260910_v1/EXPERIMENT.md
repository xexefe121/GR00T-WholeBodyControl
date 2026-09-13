# Bounded ankle feedforward fallback — complete offline lifecycle experiment

Fixed copied-state torque matrix is complete. Final1000's original rejected
state is recoverable over the checked20ms (and sampled100ms constant-target
probe) with2.5Nm inward correction, unchanged target/gains and original total
effort limits. Parent500's later boundary state remains unrecovered by all
three offsets. Neither result is full-motion or recursive-feasibility proof.

One new controller branch uses original20ms position-only range search first;
the failed100ms target-only experiment is NOT adopted. Successful original
actions and their state histories must remain bit-exact until first torque
intervention. Only after position-only bounded search rejects, reproduce its
nominal prediction and try2.5/5/10Nm corrections directed inward at the offending
ankle axes. Any non-ankle violation, both-bound crossing, or exhausted search
fails closed. Every accepted torque candidate must pass all23 original range
reserves at every2ms substep and native velocity bounds. No target/gain/limit
expansion, extra joint, external spatial force or unverified recovery command.

A separate SIM benchmark explicitly adds held feedforward BEFORE total actuator
effort clipping on each500Hz step. It records total requested/applied torque and
the separate23-vector correction; old benchmark/checkpoint sources remain
immutable. Existing policy history retains commanded bounded position-target
semantics; feedforward torque is not invented as a position target or a new
measured joint. Existing actor has no new feedforward observation or training.

Request unchanged complete PICO/002/003/008 lifecycles, original starting state
only, final1000 checkpoint ac2f844894e3e53c364ae466966a64db556ee255c6ab72e736fe15b87adb19fe.
Compare equal prefixes against the20ms final1000 and parent500 runs. Verify
zero-correction PICO prefix through the old1880-control stop exactly, then
independently replay every new actual torque, landmarks and sampled policy
decisions. Check separate torque arithmetic and recorded accepted predictions.

All original tracking, native limits, standing return, measured runtime and
pause/disconnect/recovery requirements still apply. No saved-state success or
longer source prefix qualifies live VR. No automatic correction/candidate sweep,
gain change, training extension, physical transport, mode command or export.
