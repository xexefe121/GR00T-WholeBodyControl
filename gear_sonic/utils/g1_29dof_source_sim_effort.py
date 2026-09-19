"""SIM-only source29 effort clipping; never native23 or firmware authority."""

import numpy as np

from gear_sonic.utils.g1_true23_original29_reference import SOURCE_JOINT_NAMES

BODY_ACTUATORS = tuple(name.removesuffix("_joint") for name in SOURCE_JOINT_NAMES)


def source_sim_body_effort(source_actuator_names, source_effort43):
    """Map the upstream hand-equipped scene's YAML limits by actuator name."""
    names = tuple(source_actuator_names)
    values = np.asarray(source_effort43, dtype=np.float64)
    if len(names) != 43 or len(set(names)) != 43 or values.shape != (43,):
        raise ValueError("requires 43 unique source-scene actuators and effort limits")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("source simulation efforts must be finite and positive")
    if any(name not in names for name in BODY_ACTUATORS):
        raise ValueError("source scene is missing a canonical29 body actuator")
    extras = set(names) - set(BODY_ACTUATORS)
    if len(extras) != 14 or any("_hand_" not in name for name in extras):
        raise ValueError("only the source scene's 14 hand actuators may be omitted")
    result = values[[names.index(name) for name in BODY_ACTUATORS]].copy()
    result.setflags(write=False)
    return result


class SourceSimulationEffortParameters:
    """Original C++ target/gains, separately labelled upstream SIM effort cap."""

    def __init__(self, parameters, source_actuator_names, source_effort43):
        self.base = parameters
        for name in ("default_angles", "action_scale", "kps", "kds"):
            setattr(self, name, getattr(parameters, name))
        self.effort = source_sim_body_effort(source_actuator_names, source_effort43)

    def target(self, raw29):
        return self.base.target(raw29)

    def descriptor(self):
        return dict(
            kind="source29_simulator_yaml_effort_only_v1",
            body_actuator_names=list(BODY_ACTUATORS),
            effective_sim_effort29=self.effort.tolist(),
            original_cpp_target_default_scale_kp_kd_unchanged=True,
            original29_raw_stops_unchanged=True,
            firmware_effort_verified=False,
            native23_controller_or_limits_changed=False,
            hardware_authorized=False,
            deployment_ready=False,
        )
