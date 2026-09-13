"""Deterministic scheduling/activation foundation. No controller or native runtime.

The clock and stepper are injected; only fake implementations are qualified here.
Fixed slot counts bound records, while Python record allocation is deliberately
not represented as allocation-free or as real-process timing evidence.
"""
from dataclasses import dataclass, asdict
import base64
import hashlib
import json
import math
from types import MappingProxyType
import numpy as np
from history import MeasuredHistory

STEP_NS = 2_000_000
CONTROL_STEPS = 10


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def b64(value):
    return base64.b64encode(value).decode('ascii')


@dataclass(frozen=True)
class Binding:
    run: str
    model_hash: str
    reference_hash: str
    epoch: int

    def __post_init__(self):
        if not isinstance(self.run, str) or not 0 < len(self.run) <= 128:
            raise ValueError('bounded nonempty run ID required')
        if type(self.epoch) is not int or self.epoch < 0:
            raise ValueError('nonnegative integer input epoch required')
        for value in (self.model_hash, self.reference_hash):
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('literal lowercase SHA256 identity required')


@dataclass(frozen=True)
class Command:
    command_id: str
    origin: str
    target: bytes
    raw_action: bytes

    @classmethod
    def make(cls, command_id, origin, target, raw_action):
        target, raw_action = np.asarray(target), np.asarray(raw_action)
        if target.dtype != np.float64 or raw_action.dtype != np.float32:
            raise ValueError('native target float64 and original raw action float32 required')
        if target.shape != (23,) or raw_action.shape != (23,):
            raise ValueError('23 native targets and raw actions required')
        if not np.isfinite(target).all() or not np.isfinite(raw_action).all():
            raise ValueError('nonfinite command')
        return cls(command_id, origin, target.tobytes(), raw_action.tobytes())

    def values(self):
        return np.frombuffer(self.target, dtype=np.float64).copy()

    def wire(self):
        return dict(command_id=self.command_id, origin=self.origin,
                    target=b64(self.target), raw_action=b64(self.raw_action))

    @classmethod
    def from_wire(cls, value):
        return cls(value['command_id'], value['origin'],
                   base64.b64decode(value['target'], validate=True),
                   base64.b64decode(value['raw_action'], validate=True))


@dataclass(frozen=True)
class Job:
    binding: Binding
    sequence: int
    snapshot_control: int
    snapshot_physics: int
    activation: int
    window_id: str
    snapshot_payload: bytes
    history_digest: str
    schedule_digest: str
    created_ns: int

    @property
    def input_digest(self):
        return digest(self.snapshot_payload)

    def identity(self):
        return dict(binding=asdict(self.binding), sequence=self.sequence,
                    snapshot_control=self.snapshot_control, snapshot_physics=self.snapshot_physics,
                    activation=self.activation, window_id=self.window_id, input_digest=self.input_digest,
                    history_digest=self.history_digest, schedule_digest=self.schedule_digest,
                    created_ns=self.created_ns)

    def to_bytes(self):
        return encode(dict(self.identity(), snapshot_payload=b64(self.snapshot_payload)))


@dataclass(frozen=True)
class Result:
    identity: dict
    command: Command
    completed_ns: int

    @classmethod
    def for_job(cls, job, command, completed_ns):
        return cls(job.identity(), command, completed_ns)

    def to_bytes(self):
        return encode(dict(identity=self.identity, command=self.command.wire(), completed_ns=self.completed_ns))

    @classmethod
    def from_bytes(cls, payload):
        value = json.loads(payload)
        return cls(value['identity'], Command.from_wire(value['command']), value['completed_ns'])


class CapacityFailure(RuntimeError):
    pass


class ClockFailure(RuntimeError):
    pass


class FixedLedger:
    """Append-only committed slots. Readers cannot reclaim or alter plant storage."""
    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError('negative ledger capacity')
        self._slots = [None] * capacity
        self.committed = 0
        self.overflow_record = None  # One separately reserved first-failure slot.

    def append(self, immutable_record):
        if self.committed == len(self._slots):
            if self.overflow_record is None:
                self.overflow_record = immutable_record
            raise CapacityFailure('LOGGER_CAPACITY_EXHAUSTED')
        self._slots[self.committed] = immutable_record
        self.committed += 1  # Publish after the entire payload is installed.

    def records(self):
        return tuple(self._slots[:self.committed])


@dataclass(frozen=True)
class WarningLedger:
    counts: tuple
    lastinfo: tuple


@dataclass(frozen=True)
class CapturedStep:
    simulation_time: float
    state: bytes
    torque: bytes
    warnings: WarningLedger


def owned_evidence(value):
    """Bounded typed copy of adapter returns, including invalid mutable payloads.

    Unknown objects are identified by type without invoking their repr. Large
    values retain their size/hash and explicitly marked prefix, never an alias.
    This diagnostic is not credited as a validated native capture.
    """
    def inspect(item, depth=0):
        kind = type(item)
        name = kind.__module__ + '.' + kind.__qualname__
        if depth > 8:
            return dict(type=name, truncated='depth_limit')
        if item is None or kind in (bool, int, str):
            if kind is str and len(item) > 4096:
                return dict(type=name, length=len(item), prefix=item[:4096], truncated=True)
            return dict(type=name, value=item)
        if kind is float:
            return dict(type=name, hex=item.hex())
        if kind in (bytes, bytearray):
            data = bytes(item)
            return dict(type=name, size=len(data), sha256=digest(data),
                        prefix=b64(data[:4096]), truncated=len(data)>4096)
        if kind is np.ndarray:
            return dict(type=name, dtype=item.dtype.str, shape=list(item.shape),
                        data=inspect(item.tobytes(), depth+1))
        if kind in (CapturedStep, WarningLedger):
            return dict(type=name, fields={key:inspect(getattr(item,key),depth+1)
                        for key in kind.__dataclass_fields__})
        if kind in (dict, tuple, list):
            items = list(item.items()) if kind is dict else list(enumerate(item))
            return dict(type=name, size=len(items), truncated=len(items)>64,
                        items=[[inspect(k,depth+1),inspect(v,depth+1)] for k,v in items[:64]])
        return dict(type=name, unsupported=True)
    return encode(inspect(value))


def validated_capture(value):
    """Require a fully immutable, typed adapter payload before capture credit."""
    if type(value) is not CapturedStep:
        raise ValueError('CapturedStep exact type required')
    if type(value.simulation_time) is not float or not math.isfinite(value.simulation_time) or value.simulation_time < 0:
        raise ValueError('finite nonnegative float simulation time required')
    if type(value.state) is not bytes or not 0 < len(value.state) <= 16384:
        raise ValueError('nonempty bounded immutable state bytes required')
    if type(value.torque) is not bytes or len(value.torque) != 23*8 or not np.isfinite(np.frombuffer(value.torque,np.float64)).all():
        raise ValueError('finite native23 float64 torque bytes required')
    warnings = value.warnings
    if type(warnings) is not WarningLedger:
        raise ValueError('typed WarningLedger required')
    for name, lower in [('counts',0),('lastinfo',-(2**31))]:
        fields = getattr(warnings,name)
        if type(fields) is not tuple or len(fields) != 8 or any(type(v) is not int or not lower <= v < 2**31 for v in fields):
            raise ValueError('immutable eight-int32 warning counts/lastinfo required')
    return CapturedStep(value.simulation_time,value.state,value.torque,
                        WarningLedger(warnings.counts,warnings.lastinfo))


@dataclass(frozen=True)
class StepRecord:
    index: int
    nominal_start: int
    nominal_end: int
    actual_start: int
    actual_end: int
    command_id: str
    target: bytes
    raw_action: bytes
    wake_lateness_ns: int
    finish_lateness_ns: int
    wake_debt: int
    expected_simulation_time: float
    captured: CapturedStep
    verified: bool
    issue: str | None


@dataclass(frozen=True)
class ControlRecord:
    control: int
    physics: int
    nominal_activation_ns: int
    actual_activation_ns: int
    admitted_ns: int
    nominal_window_id: str
    active_window_id: str
    command: Command
    incoming_raw: bytes
    held: bool
    history_before: tuple
    flat_history_before: bytes
    terms: tuple
    history_after: tuple
    snapshot_state: bytes


class PlantFoundation:
    def __init__(self, *, clock, stepper, epoch_ns, steps, binding, initial_command,
                 initial_previous_raw, native_limits, window_ids, results, jobs,
                 event_capacity=4096, max_debt_steps=100, max_elapsed_ns=None):
        if steps <= 0 or type(steps) is not int or type(epoch_ns) is not int:
            raise ValueError('fixed integer epoch and positive native count required')
        controls = (steps + CONTROL_STEPS - 1) // CONTROL_STEPS
        if len(window_ids) != controls or any(not isinstance(v, str) or not 0 < len(v) <= 128 for v in window_ids) or len(set(window_ids)) != controls:
            raise ValueError('one distinct preadmitted window ID for each fixed source clock required')
        self.clock, self.stepper = clock, stepper
        self._last_clock_ns = None
        self.clock_fault = None
        self.failure = None
        self.returned = 0
        self._epoch_ns, self.requested_steps, self.binding = epoch_ns, steps, binding
        self.window_ids = tuple(window_ids)
        self.results, self.jobs = results, jobs
        self.limits = np.asarray(native_limits, dtype=np.float64).copy()
        if self.limits.shape != (23, 2) or not np.isfinite(self.limits).all() or not (self.limits[:, 0] <= self.limits[:, 1]).all():
            raise ValueError('invalid native bounds')
        self.limits.flags.writeable = False
        if type(max_debt_steps) is not int or max_debt_steps < 0:
            raise ValueError('nonnegative fixed debt bound required')
        if max_elapsed_ns is not None and (type(max_elapsed_ns) is not int or max_elapsed_ns < 0):
            raise ValueError('nonnegative fixed elapsed-time bound required')
        self._validate_command(initial_command)
        if len(initial_previous_raw) != 23 * 4 or not np.isfinite(np.frombuffer(initial_previous_raw, np.float32)).all():
            raise ValueError('invalid initial raw action')
        initial_admitted_ns = self._now_ns('INITIAL_ADMISSION')
        if initial_admitted_ns >= epoch_ns:
            raise ValueError('initial command must be bound before epoch')
        self.active = initial_command
        self.active_window = self.window_ids[0]
        self.active_admitted_ns = initial_admitted_ns
        self.previous_raw = bytes(initial_previous_raw)
        self.history = MeasuredHistory()
        self.step_records, self.control_records = FixedLedger(steps), FixedLedger(controls)
        self.events = FixedLedger(event_capacity)
        self.issued = {}
        self.published_jobs = set()
        self.sealed = {}
        self.command_ids = {initial_command.command_id}
        self.attempted = self.returned = self.captured = self.verified = 0
        self.expected_simulation_time = float(stepper.initial_time)
        if not math.isfinite(self.expected_simulation_time) or self.expected_simulation_time < 0:
            raise ValueError('finite nonnegative initial simulation time required')
        self.input_fault = self.failure = self.first_deadline_failure = None
        self.max_debt_steps, self.max_elapsed_ns = max_debt_steps, max_elapsed_ns
        self.maximum_debt = self.cumulative_wake_debt = self.deadline_misses = 0
        self.stage = 'READY'
        self.last_capture = None
        self.last_capture_return = None
        self.last_verifier_return = None

    @property
    def epoch_ns(self):
        return self._epoch_ns

    def deadline(self, index):
        return self.epoch_ns + index * STEP_NS

    def _now_ns(self, phase):
        try:
            now = self.clock.now_ns()
        except Exception as exc:
            now = dict(exception=type(exc).__name__,detail=str(exc))
        if type(now) is not int or now < 0 or (self._last_clock_ns is not None and now < self._last_clock_ns):
            if self.clock_fault is None:
                self.clock_fault = MappingProxyType(dict(phase=phase, previous_ns=self._last_clock_ns,
                                                        observed=owned_evidence(now)))
            self._fail('CLOCK_NONMONOTONIC_OR_INVALID', phase=phase,
                       previous_ns=self._last_clock_ns, observed=owned_evidence(now))
            raise ClockFailure('clock must return nonnegative monotonic integer nanoseconds')
        self._last_clock_ns = now
        return now

    def debt(self, now):
        elapsed = max(0, (now - self.epoch_ns) // STEP_NS)
        return max(0, min(self.requested_steps, elapsed) - self.returned)

    def _event(self, reason, **data):
        # Immutable bytes mean a logger cannot mutate an already committed event.
        self.events.append(encode(dict(reason=reason, physics=self.returned, **data)))

    def _fail(self, reason, **data):
        if self.failure is None:
            self.failure = MappingProxyType(dict(reason=reason, physics=self.returned, **data))

    def _validate_command(self, command):
        if not isinstance(command, Command) or not isinstance(command.command_id, str) or not 0 < len(command.command_id) <= 128:
            raise ValueError('invalid command identity')
        if not isinstance(command.origin, str) or not 0 < len(command.origin) <= 128:
            raise ValueError('explicit command origin required')
        if not isinstance(command.target, bytes) or not isinstance(command.raw_action, bytes):
            raise ValueError('immutable target and raw action required')
        if len(command.target) != 23 * 8 or len(command.raw_action) != 23 * 4:
            raise ValueError('wrong command shape')
        target, raw = command.values(), np.frombuffer(command.raw_action, np.float32)
        if not np.isfinite(target).all() or not np.isfinite(raw).all():
            raise ValueError('nonfinite proposal')
        if not ((target >= self.limits[:, 0]) & (target <= self.limits[:, 1])).all():
            raise ValueError('native target outside unchanged limits')

    def poll_results(self):
        for slot, state, publication in self.results.poll_once():
            if state == 'BUSY':
                self._event('MAILBOX_BUSY', slot=slot)
            elif state == 'TAKEN':
                self._admit(publication)

    def _admit(self, publication):
        try:
            result = Result.from_bytes(publication.payload)
            identity = result.identity
            if not isinstance(identity, dict):
                raise ValueError('result identity must be an object')
            activation = identity['activation']
            reason = None
            if self.input_fault is not None:
                reason = 'FAULT_LATCHED'
            elif type(activation) is not int or publication.key != activation:
                reason = 'WRONG_ACTIVATION'
            elif encode(identity.get('binding')) != encode(asdict(self.binding)):
                reason = 'WRONG_BINDING_OR_EPOCH'
            elif activation not in self.issued:
                reason = 'UNKNOWN_JOB'
            elif activation not in self.published_jobs:
                reason = 'JOB_NOT_PUBLISHED'
            elif encode(identity) != encode(self.issued[activation].identity()):
                reason = 'JOB_OR_INPUT_MISMATCH'
            elif activation in self.sealed or result.command.command_id in self.command_ids:
                reason = 'DUPLICATE'
            else:
                self._validate_command(result.command)
                if type(result.completed_ns) is not int:
                    raise ValueError('integer worker completion timestamp required')
            # Authoritative timestamp is AFTER decoding, validation and owned copy.
            received_ns = self._now_ns('RESULT_COPY_VALIDATED')
            if reason is None and received_ns >= self.deadline(activation * CONTROL_STEPS):
                reason = 'LATE'
            if reason is None and result.completed_ns > received_ns:
                reason = 'FUTURE_WORKER_TIMESTAMP'
            if reason is None and result.completed_ns < self.issued[activation].created_ns:
                reason = 'WORKER_TIMESTAMP_BEFORE_JOB'
            self._event('RESULT_REJECTED' if reason else 'RESULT_SEALED', rejection=reason,
                        activation=activation, received_ns=received_ns,
                        worker_completed_ns=result.completed_ns, slot_version=publication.version,
                        payload=b64(publication.payload))
            if reason is None:
                self.sealed[activation] = (result.command, received_ns)
                self.command_ids.add(result.command.command_id)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self._event('RESULT_MALFORMED', payload=b64(publication.payload), detail=str(exc))

    def _boundary(self):
        control = self.returned // CONTROL_STEPS
        incoming = self.previous_raw
        held = False
        if control:
            if self.input_fault is None and control in self.sealed:
                self.active, self.active_admitted_ns = self.sealed.pop(control)
                self.active_window = self.window_ids[control]
            else:
                held = True
                if self.input_fault is None:
                    self.input_fault = MappingProxyType(dict(reason='COMMAND_DEADLINE_MISSED', control=control,
                                            nominal_deadline=self.deadline(self.returned)))
                self._event('HELD_COMMAND_INTERVAL', control=control, command_id=self.active.command_id,
                            nominal_window_id=self.window_ids[control], active_window_id=self.active_window)
        activation_ns = self._now_ns('BOUNDARY_ACTIVATION')
        measured_state, measured_terms = self.stepper.boundary_snapshot()
        if not isinstance(measured_state, bytes):
            raise ValueError('owned immutable boundary state required')
        before, flat, terms, after = self.history.advance(measured_terms, incoming)
        record = ControlRecord(control, self.returned, self.deadline(self.returned), activation_ns,
                               self.active_admitted_ns, self.window_ids[control], self.active_window,
                               self.active, incoming, held, before, flat, terms, after, measured_state)
        self.control_records.append(record)
        self.previous_raw = self.active.raw_action
        if control + 1 < len(self.window_ids) and self.input_fault is None:
            payload = encode(dict(state=b64(measured_state), incoming_raw=b64(incoming),
                active_command=self.active.wire(), history_before=[(k,b64(v)) for k,v in before],
                terms=[(k,b64(v)) for k,v in terms], history_after=[(k,b64(v)) for k,v in after]))
            job = Job(self.binding, control, control, self.returned, control + 1,
                      self.window_ids[control + 1], payload, digest(flat), digest(encode(self.active.wire())),
                      self._now_ns('JOB_CREATED'))
            self.issued[control + 1] = job
            publication = self.jobs.try_publish(control + 1, job.to_bytes())
            if publication == 'PUBLISHED':
                self.published_jobs.add(control + 1)
            self._event('JOB_PUBLICATION', activation=control + 1, status=publication,
                        source_window=job.window_id, input_digest=job.input_digest)

    def tick(self):
        if self.failure is not None or self.returned >= self.requested_steps:
            return False
        try:
            self.poll_results()  # Opportunistic, outside the sealed activation switch.
            if self.failure is not None:
                return False
            index = self.returned
            self.stage = 'WAITING_FOR_FIXED_DEADLINE'
            self.clock.wait_until_ns(self.deadline(index))
            start = self._now_ns('STEP_WAKE')
            if type(start) is not int or start < self.deadline(index):
                self._fail('CLOCK_RETURNED_BEFORE_FIXED_DEADLINE')
                return False
            wake_debt = self.debt(start)
            self.maximum_debt = max(self.maximum_debt, wake_debt)
            self.cumulative_wake_debt += wake_debt
            if start > self.deadline(index + 1) and self.first_deadline_failure is None:
                self.first_deadline_failure = MappingProxyType(dict(index=index, nominal_end=self.deadline(index + 1),
                    observed_ns=start, phase='WAKE_ALREADY_OVERDUE', lateness_ns=start-self.deadline(index+1)))
            if wake_debt > self.max_debt_steps:
                self._fail('DEBT_BOUND_EXCEEDED', debt=wake_debt)
                return False
            if self.max_elapsed_ns is not None and start - self.epoch_ns > self.max_elapsed_ns:
                self._fail('RUN_BOUND_EXCEEDED')
                return False
            if index % CONTROL_STEPS == 0:
                self.stage = 'BOUNDARY_ACTIVATION_AND_HISTORY'
                self._boundary()
            self.stage = 'STEP_ATTEMPT'
            self.last_capture = None
            self.last_capture_return = None
            self.last_verifier_return = None
            self.attempted += 1
            self.stepper.step(self.active.values())
            self.returned += 1  # A returned strict failure still owns its actual step.
            self.expected_simulation_time += 0.002
            self.stage = 'STEP_CAPTURE'
            snapshot = self.stepper.capture_step()
            self.last_capture_return = owned_evidence(snapshot)
            snapshot = validated_capture(snapshot)
            self.last_capture = snapshot
            self.captured += 1
            self.stage = 'STEP_VERIFY'
            issue = self.stepper.verify_step(snapshot)
            self.last_verifier_return = owned_evidence(issue)
            if issue is not None and (type(issue) is not str or not 0 < len(issue) <= 1024):
                raise ValueError('verifier must return None or a nonempty bounded string')
            if snapshot.simulation_time != self.expected_simulation_time:
                issue = issue or 'SIMULATION_CLOCK_MISMATCH'
            if issue is None:
                self.verified += 1
            finish = self._now_ns('STEP_FINISH')
            self.maximum_debt = max(self.maximum_debt, self.debt(finish))
            lateness = max(0, finish - self.deadline(index + 1))
            if lateness:
                self.deadline_misses += 1
                if self.first_deadline_failure is None:
                    self.first_deadline_failure = MappingProxyType(dict(index=index, nominal_end=self.deadline(index + 1),
                                                       actual_end=finish, lateness_ns=lateness))
            self.step_records.append(StepRecord(index, self.deadline(index), self.deadline(index + 1),
                start, finish, self.active.command_id, self.active.target, self.active.raw_action,
                max(0, start - self.deadline(index)), lateness, wake_debt,
                self.expected_simulation_time, snapshot, issue is None, issue))
            if issue is not None:
                self._fail('STRICT_STEP_VIOLATION', issue=issue)
            self.stage = 'READY' if self.failure is None else 'FAILED'
            return self.failure is None
        except Exception as exc:
            self._fail('LOGGER_CAPACITY_EXHAUSTED' if isinstance(exc, CapacityFailure) else 'STEPPER_OR_CAPTURE_EXCEPTION',
                       exception=type(exc).__name__, detail=str(exc), stage=self.stage,
                       mutation_uncertain=self.attempted > self.returned)
            self.stage = 'FAILED'
            return False

    def summary(self):
        try:
            now = self._now_ns('SUMMARY')
            current_debt = self.debt(now)
        except ClockFailure:
            current_debt = None  # Invalid current time has no honest debt value.
        return dict(requested=self.requested_steps, attempted=self.attempted, returned=self.returned,
            captured=self.captured, verified=self.verified, committed=self.step_records.committed,
            unexecuted=self.requested_steps-self.returned, current_debt=current_debt,
            last_valid_clock_ns=self._last_clock_ns,
            clock_fault=dict(self.clock_fault) if self.clock_fault is not None else None,
            max_debt=self.maximum_debt, cumulative_wake_debt=self.cumulative_wake_debt,
            deadline_misses=self.deadline_misses, first_deadline_failure=dict(self.first_deadline_failure) if self.first_deadline_failure is not None else None,
            epoch_ns=self.epoch_ns, failure=dict(self.failure) if self.failure is not None else None,
            input_fault=dict(self.input_fault) if self.input_fault is not None else None,
            history_entries=self.history.entries,
            timing_passed=self.returned == self.requested_steps and self.first_deadline_failure is None and self.failure is None,
            physical_qualification=False, real_process_timing_qualified=False)
