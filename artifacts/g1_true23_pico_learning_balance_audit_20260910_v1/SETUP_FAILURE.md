# First audit setup failure preserved

analyze.py exited1 before reading data: SONIC_HARDWARE_DEFAULT_Q lives in
gear_sonic.envs.mjlab.sonic_true23, not g1_23dof_contract. No audit result,
training, dynamics, policy or physical command ran. Original script retained.
analyze_v2.py corrects that import and independently checks latest-history
joint-position/velocity decoding against existing measured PICO states before
using the decoder for training-state coverage. Float32 position reconstruction
tolerance5e-7rad is diagnostic arithmetic only, not a physical acceptance gate.
