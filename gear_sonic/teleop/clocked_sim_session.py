"""Fixed-rate SIM input handling; a missing packet never pauses physics.

The caller supplies consecutive 20-ms virtual deadlines. This module does not
open a transport or prove wall-clock schedulability. Missing/invalid input
latches the existing balance policy on that tick. Later packets cannot silently
restart SONIC. A deliberate reacquisition controller remains separate work.
"""

import numpy as np

from gear_sonic.utils.g1_true23_frozen_lora_live_teleop import (
    CONTROL_PERIOD_NS,
    LiveTransportFault,
    step_live_packet,
    validate_live_packet,
)


class ClockedSimSession:
    def __init__(self, controller, *, start_ns, maximum_age_ns=100_000_000):
        if type(start_ns) is not int or start_ns < 0:
            raise ValueError("start must be a nonnegative integer timestamp")
        if type(maximum_age_ns) is not int or not CONTROL_PERIOD_NS <= maximum_age_ns <= 100_000_000:
            raise ValueError("packet freshness must remain between 20 and 100 ms")
        if controller.completed != 0 or abs(float(controller.data.time)) > 1e-12:
            raise ValueError("session requires an explicitly initialized fresh simulation")
        self.controller = controller
        self.start_ns = start_ns
        self.maximum_age_ns = maximum_age_ns
        self.ticks = 0
        self.previous = None
        self.fault = None
        self.fault_tick = None
        self.fault_detail = None
        self.stopped = False
        self.sonic_controls = 0
        self.packets_ignored_after_latch = 0

    def tick(self, packet, *, deadline_ns):
        if self.stopped:
            raise RuntimeError("failed simulation cannot be resumed implicitly")
        if type(deadline_ns) is not int or deadline_ns != self.start_ns + self.ticks * CONTROL_PERIOD_NS:
            raise ValueError("simulation deadlines must be contiguous at exactly 50 Hz")
        controller = self.controller
        before_count, before_time = controller.completed, float(controller.data.time)
        summary = None
        if not controller.fallback_active:
            try:
                if packet is None:
                    raise LiveTransportFault("timeout", "no source packet at control deadline")
                summary = validate_live_packet(
                    packet, previous=self.previous, maximum_age_ns=self.maximum_age_ns, now_ns=deadline_ns
                )
            except LiveTransportFault as failure:
                self.fault, self.fault_tick, self.fault_detail = failure.fault, self.ticks, str(failure)
                controller.activate_fallback(failure.trigger)
        elif packet is not None:
            self.packets_ignored_after_latch += 1
        self.ticks += 1
        try:
            if controller.fallback_active:
                evidence = controller.step(np.zeros(267, dtype=np.float32))
            else:
                evidence = step_live_packet(controller, packet)
                self.previous = summary
                if not controller.fallback_active:
                    self.sonic_controls += 1
            if (
                controller.completed != before_count + 1
                or abs(float(controller.data.time) - before_time - 0.02) > 1e-8
            ):
                raise RuntimeError("one control deadline must integrate exactly 20 ms")
            return dict(evidence, deadline_ns=deadline_ns, input_fault=self.fault)
        except RuntimeError:
            self.stopped = True
            raise


class SubstepObservation:
    """Instance-only MuJoCo forwarding proxy; no altered return or global patch."""

    def __init__(self, module):
        self.module = module
        self.joints = []
        self.velocities = []
        self.torques = []
        self.times = []

    def __getattr__(self, name):
        return getattr(self.module, name)

    def mj_step(self, model, data):
        self.torques.append(data.ctrl.copy())
        result = self.module.mj_step(model, data)
        self.joints.append(data.qpos[7:].copy())
        self.velocities.append(data.qvel[6:].copy())
        self.times.append(float(data.time))
        return result

    def arrays(self):
        return dict(
            physics_joint_pos=np.asarray(self.joints, dtype=np.float64).reshape(-1, 23),
            physics_joint_vel=np.asarray(self.velocities, dtype=np.float64).reshape(-1, 23),
            physics_command_torque=np.asarray(self.torques, dtype=np.float64).reshape(-1, 23),
            physics_time=np.asarray(self.times, dtype=np.float64),
        )
