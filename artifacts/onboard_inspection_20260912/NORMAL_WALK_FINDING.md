# Separate factory normal-walk policy

Local public firmware contains five `normal_walk/*.mnn` models. Each has a
49-value observation and 13 outputs for the legs and waist yaw. The selected
configuration is `G1_Normal_Walk_20240827_103556_22000.mnn`, with a 0.01s control
period and one observation-history sample. The configuration also contains a
23-joint branch, but none of these five shipped networks has23 outputs.

`FsmNormalWalk::ProcessLocoCmd` consists only of `mov w0,#1; ret`. Its observation
routine applies gyro and joint-velocity filtering, advances a cadence counter,
includes three zero-valued observation slots, and explicitly zeros four ankle
velocity entries. The model contract cannot be inferred from the tensor shape
alone; those three slots must not simply be assumed to accept velocity commands.
No direct received full-body pose input was found in this callback. This does
not establish that the model cannot be retrained; it does rule out treating
the recovered callback as an already implemented full-body teleop controller.

No native rollout or model training was run for this profile. Its100Hz factory
controller also differs from the requested50Hz interface. Existing models and
launchers remain unchanged.

Evidence in `onboard_factory_firmware_v1`: `mnn_graph_catalog.json`,
`decoded_configs/policies/normal_walk/fsm_cfg.yaml`,
`normal_walk13_disassembly.txt`, `normal_walk13_init.txt`, and
`normal_walk_command_handler.txt`. ARM instructions were inspected locally;
no robot program, service, motor command or network configuration was changed.
