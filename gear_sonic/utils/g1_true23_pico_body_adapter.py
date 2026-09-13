"""Current Pico/SOMA body geometry into the existing received-body interface.

The producer and consumer must share a monotonic clock (same Linux host).
Source timestamps are retained; an affine epoch offset never refreshes a stale
packet. Old delayed IL23 payloads are deliberately insufficient for this path.
No socket, robot feedback, policy, forward prediction, or motor publication.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_pico_retargeted_producer import SOMA_MJ29_JOINT_NAMES

# Native23 changes the terminal roll body's mesh/name; the joint-frame origin
# remains the source wrist-roll frame. Original wrist-yaw hand goals below are
# separate task points and must not be replaced with these body origins.
SOURCE_BODY_ALIASES = {
    'left_wrist_roll_rubber_hand': 'left_wrist_roll_link',
    'right_wrist_roll_rubber_hand': 'right_wrist_roll_link',
}


@dataclass(frozen=True)
class AdaptedBodyPacket:
    packet: object
    task_position: np.ndarray
    task_quaternion: np.ndarray
    source: dict

    def controller_args(self):
        return self.packet, self.task_position, self.task_quaternion


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def _array(value, shape, name):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite shape {shape}')
    return result


def _quaternions(value, count, name):
    result = _array(value, (count, 4), name)
    if np.max(np.abs(np.linalg.norm(result, axis=1)-1)) > 1e-5:
        raise ValueError(f'{name} must be normalized XYZW')
    return result[:, [3, 0, 1, 2]]


class Native23PicoBodyAdapter:
    def __init__(self, contract, *, epoch=0, maximum_age_ns=100_000_000):
        self.body_names = tuple(contract['body_names'])
        if len(self.body_names) != 24 or self.body_names[0] != 'pelvis':
            raise ValueError('native body contract must contain 24 bodies, pelvis first')
        if not 0 < maximum_age_ns <= 100_000_000:
            raise ValueError('adapter cannot weaken the 100 ms packet deadline')
        self.maximum_age_ns = maximum_age_ns
        self.epoch = _integer(epoch, 'epoch')
        self.fault = None
        self.previous = None
        self.origin_ns = self.origin_frame = None

    def rearm(self, epoch):
        """Explicit input rearm; caller must also rearm the controller gate."""
        if _integer(epoch, 'epoch') != self.epoch + 1:
            raise ValueError('rearm must advance exactly one controller epoch')
        self.epoch = epoch
        self.fault = self.previous = self.origin_ns = self.origin_frame = None

    def adapt(self, wire, *, received_monotonic_ns):
        if self.fault is not None:
            raise ValueError(f'Pico adapter fault latched; explicit rearm required: {self.fault}')
        try:
            return self._adapt(wire, received_monotonic_ns)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            self.fault = str(exc)
            raise ValueError(f'Pico body packet rejected: {exc}') from exc

    def _adapt(self, wire, received_ns):
        from gear_sonic.utils.g1_true23_bfmzero_stream import Packet, validate_fields
        body = wire['native23_body_pose']
        if (body['schema_version'] != 1 or body['kind'] != 'native23_current_original29_body_pose'
                or body['original29_axes_preserved'] is not True):
            raise ValueError('current original29 full-body extension required')
        received_ns = _integer(received_ns, 'receive timestamp')
        stamp = _integer(body['reference_monotonic_ns'], 'reference timestamp')
        capture = _integer(body['capture_monotonic_ns'], 'capture timestamp')
        frame = _integer(body['source_frame_index'], 'source frame')
        if stamp != wire['control_monotonic_ns'] or frame != wire['control_source_frame_index']:
            raise ValueError('body geometry does not belong to current transport frame')
        age = received_ns-stamp
        if age < 0 or age > self.maximum_age_ns:
            raise ValueError('source timestamp is future or stale on declared shared clock')
        if capture > received_ns or received_ns-capture > self.maximum_age_ns:
            raise ValueError('capture timestamp is future or stale')
        names = tuple(body['joint_names'])
        if names != SOMA_MJ29_JOINT_NAMES:
            raise ValueError('original29 named joint order mismatch')
        all_q = _array(body['joint_position'], (29,), 'original29 joint position')
        q = all_q[[names.index(name) for name in HARDWARE_23_JOINT_NAMES]]
        names = tuple(body['body_names'])
        mapped_names = tuple(name if name in names else SOURCE_BODY_ALIASES.get(name,name)
                             for name in self.body_names)
        if len(set(names)) != len(names) or any(name not in names for name in mapped_names):
            raise ValueError('source lacks uniquely named native body objectives')
        if any(native in names and source in names for native,source in SOURCE_BODY_ALIASES.items()):
            raise ValueError('ambiguous native/source wrist body aliases')
        indices = [names.index(name) for name in mapped_names]
        pos = _array(body['body_position_w'], (len(names), 3), 'body position')[indices]
        quat = _quaternions(body['body_quaternion_xyzw'], len(names), 'body quaternion')[indices]
        if body['task_names'] != ['left_hand', 'right_hand', 'head']:
            raise ValueError('all three original hand/head objectives required')
        tasks = _array(body['task_position_w'], (3, 3), 'hand/head position')
        task_quat = _quaternions(body['task_quaternion_xyzw'], 3, 'hand/head quaternion')
        dq, linear, angular = np.zeros(23), np.zeros((24, 3)), np.zeros((24, 3))
        previous = self.previous
        if previous is not None:
            if frame != previous['frame']+1 or stamp != previous['stamp']+20_000_000:
                raise ValueError('noncontiguous received 50 Hz poses')
            if received_ns < previous['received'] or capture < previous['capture']:
                raise ValueError('receive or capture clock regressed')
            dq = (q-previous['q'])/.02
            linear = (pos-previous['pos'])/.02
            current = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
            prior = Rotation.from_quat(previous['quat'][:, [1, 2, 3, 0]])
            angular = (current*prior.inv()).as_rotvec()/.02
        else:
            self.origin_ns, self.origin_frame = stamp, frame
        fields = dict(joint_pos=q, joint_vel=dq, body_pos_w=pos, body_quat_w=quat,
                      body_lin_vel_w=linear, body_ang_vel_w=angular)
        validate_fields(fields)
        sequence = frame-self.origin_frame
        packet = Packet(self.epoch, sequence, (stamp-self.origin_ns)/1e9, fields)
        source = dict(reference_monotonic_ns=stamp, capture_monotonic_ns=capture,
                      source_frame_index=frame, received_monotonic_ns=received_ns,
                      source_age_ns=age, source_origin_ns=self.origin_ns,
                      derivative='received_current_minus_received_previous_20ms',
                      first_derivative_unavailable=previous is None,
                      body_name_mapping=dict(zip(self.body_names,mapped_names)),
                      clock_domain='shared_host_monotonic', original29_axes_preserved=True)
        self.previous = dict(frame=frame, stamp=stamp, received=received_ns, capture=capture,
                             q=q.copy(), pos=pos.copy(), quat=quat.copy())
        return AdaptedBodyPacket(packet, tasks, task_quat, source)

    def receive(self, controller, wire, *, received_monotonic_ns, now):
        """Admit through the existing packet gate, retaining real source age."""
        try:
            result = self.adapt(wire, received_monotonic_ns=received_monotonic_ns)
            gate = controller.receiver.gate
            if gate.epoch != self.epoch:
                raise ValueError('adapter/controller rearm epochs differ')
            if result.packet.sequence == 0:
                if gate.last_sequence != -1:
                    raise ValueError('cannot replace an active receiver epoch')
                gate.epoch_started_at = float(now)-(received_monotonic_ns-self.origin_ns)/1e9
            if not controller.receive(*result.controller_args(), now):
                raise ValueError(f'controller admission rejected: {gate.fault}')
            return result
        except ValueError as exc:
            self.fault = str(exc)
            controller.receiver.gate.latch('pico_body_adapter:'+str(exc), now)
            raise
