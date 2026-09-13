# Native23 learning/balance audit — fixed saved data, no new training

Previous goal turn made progress: implemented and rejected two complete
controller matrices, verified a local ankle correction and exposed a later
whole-body fall. All jobs are terminal. Guard changes alone are not a solution.

Audit the existing parent500 and foot-continuation1000 training captures before
changing learning. Original-intent lower-body posture is still penalized while
original29 hand/head tasks remain targets; the presence of both is not proof
of a harmful conflict. Actual perturbations already exist. An earlier fresh
failure-weighted sampler also failed; do not present it as an untried remedy.

Use only existing hash-bound training rewards, full training failure/reference
captures, sampled actual267/930/9 inputs, the original immutable bank and the
saved final1000/ankle-feedforward PICO trajectories. No physics, optimization,
retargeting, guard, source, model, gain or acceptance change in this audit.

Report per-term raw cost, bounded negative reward contribution and transform
slope, distinguishing reward units from physical/PPO gradients. Keep post-reset
states out of nonterminal reward summaries. Count actual phase exposure using
full captured reference indices, not sparse observation samples alone. Require
exact phase-index agreement at sampled control boundaries before joining data.

Reconstruct current23 joint positions/velocities from the stored noiseless
joint-observation channels with the original permutations/default, explicitly
allowing float32 subtraction/addition roundoff. Use uncorrupted captured root
quaternion and Root9 for height/tilt/speed/error features. Do not reconstruct
exact gyro/gravity from their deliberately noisy channels. Compare the failing
PICO phase against marginal training-state distributions within+/-25frames
and report coverage counts. Marginal overlap is not joint-distribution coverage
or proof that a state is in/out of distribution. Changing training populations
also precludes a causal learning-curve claim.

Only a measured finding that changes the next action justifies further work.
No automatic reset/reward/optimizer sweep or continuation. Full-body source
fidelity, both legs, standing return, timing/recovery and later supervised live
checks remain required. No physical robot transport, commands or export.
