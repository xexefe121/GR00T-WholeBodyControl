"""Deterministic virtual-clock tests only; no spawned/native/policy runtimes."""
from dataclasses import replace
import ast
import json
from pathlib import Path
import numpy as np
import pytest
from clock_core import (Binding, Command, Result, CapturedStep, WarningLedger, PlantFoundation,
                        STEP_NS, FixedLedger, CapacityFailure, encode, digest)
from history import MeasuredHistory, SIZES
from mailbox import Mailbox, ByteSlot


class TryLock:
    def __init__(self):
        self.held = False
        self.attempts = []

    def acquire(self, blocking):
        assert blocking is False
        self.attempts.append(blocking)
        if self.held:
            return False
        self.held = True
        return True

    def release(self):
        assert self.held
        self.held = False


class Clock:
    def __init__(self):
        self.ns = 0
        self.waits = []

    def now_ns(self):
        return self.ns

    def wait_until_ns(self, deadline):
        self.waits.append(deadline)
        self.ns = max(self.ns, deadline)

    def advance(self, duration):
        self.ns += duration


class Stepper:
    initial_time = 0.0

    def __init__(self, clock, *, duration=100_000, thrown=None, capture_thrown=None,
                 strict=None, verify_thrown=None):
        self.clock, self.duration = clock, duration
        self.thrown, self.capture_thrown, self.strict, self.verify_thrown = thrown, capture_thrown, strict, verify_thrown
        self.count = 0
        self.time = 0.0
        self.targets = []

    def boundary_snapshot(self):
        return np.array([self.count], np.int64).tobytes(), {
            key: np.full(size, self.count, np.float32) for key, size in SIZES.items() if key != 'actions'}

    def step(self, target):
        if self.count == self.thrown:
            raise RuntimeError('scripted step exception')
        self.targets.append(target.copy())
        self.count += 1
        self.time += 0.002
        self.clock.advance(self.duration)

    def capture_step(self):
        if self.count == self.capture_thrown:
            raise RuntimeError('scripted capture exception')
        return CapturedStep(self.time, np.array([self.count], np.int64).tobytes(),
                            self.targets[-1].tobytes(), WarningLedger((0,) * 8, (0,) * 8))

    def verify_step(self, captured):
        if self.count == self.verify_thrown:
            raise RuntimeError('scripted verification exception')
        return 'SCRIPTED_STRICT_VIOLATION' if self.count == self.strict else None


def command(name='initial', raw=5., target=1.):
    return Command.make(name, 'SCRIPTED_RAW_PROPOSAL', np.full(23, target, np.float64), np.full(23, raw, np.float32))

def plant(steps=60, **kwargs):
    clock = Clock()
    stepper = Stepper(clock, **kwargs.pop('stepper', {}))
    windows=kwargs.pop('window_ids',[f'window-{k}' for k in range((steps+9)//10)])
    p = PlantFoundation(clock=clock, stepper=stepper, epoch_ns=100_000_000,
        steps=steps, binding=Binding('fake-run', digest(b'fake-model'), digest(b'fake-reference'), 3),
        initial_command=command(), initial_previous_raw=np.zeros(23, np.float32).tobytes(),
        native_limits=np.tile([-1., 1.], (23, 1)), window_ids=windows,
        results=Mailbox(3, 32768, TryLock), jobs=Mailbox(3, 32768, TryLock), **kwargs)
    return p

def run(p):
    while p.tick():
        pass
    return p.summary()

def publish(p, activation, candidate=None, *, completed=None, identity=None):
    job = p.issued[activation]
    result = Result.for_job(job, candidate or command(f'command-{activation}', raw=activation+5.),
                           p.clock.now_ns() if completed is None else completed)
    if identity is not None:
        result = replace(result, identity=identity)
    assert p.results.try_publish(activation, result.to_bytes()) == 'PUBLISHED'
    return result

def reasons(p):
    return [json.loads(v) for v in p.events.records()]


def test_hung_worker_100ms_and_initial_command_hold():
    p = plant(50)
    # Worker input wait and output lock remain held indefinitely; plant never waits.
    p.jobs.slots[1].lock.held = True
    p.results.slots[2].lock.held = True
    summary = run(p)
    assert summary['returned'] == summary['captured'] == summary['verified'] == 50
    assert summary['timing_passed'] and summary['max_debt'] == 0
    rows = p.step_records.records()
    assert [r.index for r in rows] == list(range(50))
    assert rows[-1].nominal_end - p.epoch_ns == 100_000_000
    assert all(r.command_id == 'initial' for r in rows)
    controls = p.control_records.records()
    assert [r.control for r in controls] == list(range(5)) and p.history.entries == 5
    assert [r.held for r in controls] == [False, True, True, True, True]
    assert [r.nominal_window_id for r in controls] == [f'window-{k}' for k in range(5)]
    assert all(r.active_window_id == 'window-0' for r in controls)
    assert any(e['reason'] == 'MAILBOX_BUSY' for e in reasons(p))
    assert any(e.get('status') == 'BUSY' for e in reasons(p))


def test_early_result_waits_for_exact_boundary():
    p = plant(30)
    assert p.tick()
    selected = command('selected', raw=42., target=.5)
    publish(p, 1, selected)
    for _ in range(9):
        assert p.tick()
    assert p.returned == 10 and all(r.command_id == 'initial' for r in p.step_records.records())
    assert p.tick()
    assert p.step_records.records()[-1].command_id == 'selected'
    assert p.control_records.records()[1].incoming_raw == command().raw_action
    assert p.control_records.records()[1].command.raw_action == selected.raw_action
    activation=p.control_records.records()[1]
    assert activation.admitted_ns < activation.nominal_activation_ns == activation.actual_activation_ns


def test_identical_clipped_targets_distinct_raw_and_rejected_history():
    p = plant(40)
    p.tick()
    activated, rejected = command('activated', raw=12.), command('rejected', raw=99.)
    assert activated.target == rejected.target and activated.raw_action != rejected.raw_action
    publish(p, 1, activated)
    p.tick()  # seal first bundle
    history_before_rejection = p.history.snapshot()
    publish(p, 1, rejected)
    p.tick()
    assert p.history.snapshot() == history_before_rejection
    summary = run(p)
    assert summary['returned'] == 40 and summary['history_entries'] == 4
    controls = p.control_records.records()
    assert controls[1].command == activated
    assert controls[2].held and controls[2].command == activated
    assert controls[3].held and controls[3].incoming_raw == activated.raw_action
    # No inverse-clipped-target reconstruction: actions contain the selected raw12.
    named = dict(controls[2].terms)
    assert named['actions'] == activated.raw_action
    assert any(e.get('rejection') == 'DUPLICATE' for e in reasons(p))


@pytest.mark.parametrize('field,value,expected', [
    ('activation', 2, 'WRONG_ACTIVATION'), ('sequence', 99, 'JOB_OR_INPUT_MISMATCH'),
    ('snapshot_physics', 1, 'JOB_OR_INPUT_MISMATCH'), ('snapshot_control', 1, 'JOB_OR_INPUT_MISMATCH'),
    ('window_id', 'wrong-source-clock', 'JOB_OR_INPUT_MISMATCH'),
    ('history_digest', 'bad', 'JOB_OR_INPUT_MISMATCH'), ('input_digest', 'bad', 'JOB_OR_INPUT_MISMATCH'),
    ('schedule_digest', 'bad', 'JOB_OR_INPUT_MISMATCH')])
def test_wrong_result_identity_never_seals(field, value, expected):
    p = plant();p.tick()
    identity = p.issued[1].identity();identity[field] = value
    publish(p, 1, identity=identity);p.tick()
    assert not p.sealed
    assert any(e.get('rejection') == expected for e in reasons(p))


@pytest.mark.parametrize('epoch', [2, 4, 3.0])
def test_old_future_and_wrong_type_epoch_rejected(epoch):
    p = plant();p.tick()
    identity = p.issued[1].identity();identity['binding']['epoch'] = epoch
    publish(p, 1, identity=identity);p.tick()
    assert not p.sealed
    assert any(e.get('rejection') == 'WRONG_BINDING_OR_EPOCH' for e in reasons(p))


def test_worker_timestamp_cannot_backdate_delayed_publication():
    p = plant(40);p.tick()
    old_completed = p.clock.now_ns()
    # Keep plant schedulable for30ms; worker has not published despite old timestamp.
    for _ in range(15):p.tick()
    publish(p, 1, completed=old_completed);p.tick()
    assert p.input_fault is not None
    event = [e for e in reasons(p) if e['reason'] == 'RESULT_REJECTED'][-1]
    assert event['received_ns'] > old_completed + 20_000_000
    assert event['rejection'] == 'FAULT_LATCHED'
    assert all(r.command_id == 'initial' for r in p.step_records.records())


def test_copy_validation_completion_at_deadline_is_late():
    p = plant();p.tick()
    publish(p, 1)
    # Deadline passes between publication and plant copy/validation completion.
    p.clock.ns = p.deadline(10)
    p.poll_results()
    assert not p.sealed
    assert any(e.get('rejection') == 'LATE' and e['received_ns'] == p.deadline(10) for e in reasons(p))


def test_future_or_before_job_worker_timestamp_rejected():
    for timestamp_offset,expected in [(1, 'FUTURE_WORKER_TIMESTAMP'), (-1_000_000, 'WORKER_TIMESTAMP_BEFORE_JOB')]:
        p = plant();p.tick()
        publish(p, 1, completed=p.clock.now_ns()+timestamp_offset)
        p.poll_results()
        assert not p.sealed and any(e.get('rejection') == expected for e in reasons(p))


def test_malformed_payload_rejected_without_stopping_steps():
    p=plant();p.tick()
    for payload in (b'not-json', encode(dict(identity=[],command=command().wire(),completed_ns=0))):
        assert p.results.try_publish(1,payload)=='PUBLISHED'
        assert p.tick()
    assert p.failure is None and not p.sealed
    assert sum(e['reason']=='RESULT_MALFORMED' for e in reasons(p))==2


def test_23ms_plant_stall_keeps_every_index_and_deadline_failure():
    p=plant(50);p.tick();original_epoch=p.epoch_ns
    p.clock.advance(23_000_000)
    p.tick();first=dict(p.first_deadline_failure)
    summary=run(p)
    assert summary['returned']==50 and summary['current_debt']==0
    assert summary['max_debt']>=10 and summary['cumulative_wake_debt']>=10
    assert summary['first_deadline_failure']==first and not summary['timing_passed']
    assert p.epoch_ns==original_epoch
    assert [r.index for r in p.step_records.records()]==list(range(50))
    assert p.clock.waits==[original_epoch+n*STEP_NS for n in range(50)]
    with pytest.raises(AttributeError):p.epoch_ns=999


def test_debt_bound_preserves_exact_unexecuted_prefix():
    p=plant(50,max_debt_steps=3);p.tick();p.clock.advance(23_000_000)
    assert not p.tick()
    s=p.summary()
    assert s['returned']==s['attempted']==1 and s['unexecuted']==49
    assert s['current_debt']>=10 and s['first_deadline_failure'] is not None
    assert s['failure']['reason']=='DEBT_BOUND_EXCEEDED' and not s['timing_passed']


def test_run_bound_does_not_rebase_or_skip():
    p=plant(50,max_elapsed_ns=5_000_000)
    s=run(p)
    assert s['returned']==3 and s['unexecuted']==47 and s['failure']['reason']=='RUN_BOUND_EXCEEDED'
    assert p.clock.waits[-1]==p.epoch_ns+3*STEP_NS


@pytest.mark.parametrize('settings,counts,stage', [
    ({'thrown':2},(3,2,2,2,2),'STEP_ATTEMPT'),
    ({'capture_thrown':3},(3,3,2,2,2),'STEP_CAPTURE'),
    ({'verify_thrown':3},(3,3,3,2,2),'STEP_VERIFY'),
    ({'strict':3},(3,3,3,2,3),None)])
def test_step_attempt_return_capture_verify_are_distinct(settings,counts,stage):
    p=plant(20,stepper=settings);s=run(p)
    assert tuple(s[k] for k in ('attempted','returned','captured','verified','committed'))==counts
    assert s['unexecuted']==20-counts[1]
    expected=0.
    for _ in range(counts[1]):expected+=.002
    assert p.expected_simulation_time==expected
    if stage:
        assert s['failure']['stage']==stage
        assert s['failure']['mutation_uncertain']==(counts[0]>counts[1])
    else:
        assert s['failure']['reason']=='STRICT_STEP_VIOLATION'
        assert p.step_records.records()[-1].verified is False
    if stage=='STEP_VERIFY':assert p.last_capture.simulation_time==expected


def test_repeated_simulation_time_not_integer_schedule_replacement():
    p=plant(200);s=run(p)
    expected=0.
    for _ in range(200):expected+=.002
    assert p.expected_simulation_time==expected and expected!=200*.002
    assert s['verified']==200


def test_logger_blockage_does_not_require_draining_reserved_trace():
    p=plant(100)
    # Logger never reads anything while plant executes its full reserved run.
    s=run(p)
    assert s['returned']==s['committed']==100 and s['failure'] is None
    assert len(p.step_records.records())==100


def test_logger_capacity_failure_never_blocks_and_retains_overflow():
    p=plant(20,event_capacity=0)
    assert not p.tick()
    s=p.summary()
    assert s['failure']['reason']=='LOGGER_CAPACITY_EXHAUSTED'
    assert s['attempted']==s['returned']==0 and s['unexecuted']==20
    assert p.events.committed==0 and p.events.overflow_record is not None
    assert json.loads(p.events.overflow_record)['reason']=='JOB_PUBLICATION'


def test_history_matches_original_operation_and_signed_zero_bytes():
    history=MeasuredHistory()
    source=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/one_step_physical_student_v1/source_snapshot_v1/gear_sonic/utils/g1_true23_bfm_seed_observations.py')
    assert digest(source.read_bytes())=='a3c68a9aefad87c3dbda5b515f34a968923e9d35234a278b01b727d73062f68d'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='BFMHistory')
    scope={'np':np}
    # Only the original pure history class executes; no source-module imports.
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    original=scope['BFMHistory']()
    for k in range(15):
        raw=np.full(23,k,np.float32);raw[0]=-0.0
        terms={key:np.full(size,k+.5,np.float32) for key,size in SIZES.items() if key!='actions'}
        expected=original.before_update(dict(terms,actions=raw))
        _,flat,_,after=history.advance(terms,raw.tobytes())
        assert flat==expected.tobytes()
        assert dict(after)=={key:original.data[key].tobytes() for key in original.data}
    assert history.entries==15


def test_invalid_history_term_does_not_partially_commit():
    h=MeasuredHistory();before=h.snapshot()
    terms={key:np.zeros(size,np.float32) for key,size in SIZES.items() if key!='actions'}
    terms['dof_vel'][0]=np.nan
    with pytest.raises(ValueError):h.advance(terms,np.zeros(23,np.float32).tobytes())
    assert h.snapshot()==before and h.entries==0


def test_mailbox_nonblocking_owned_copy_versions_and_no_overwrite():
    m=Mailbox(1,16,TryLock)
    assert m.try_publish(1,b'first')=='PUBLISHED'
    assert m.try_publish(2,b'newer')=='FULL'
    _,state,first=m.poll_once()[0]
    assert state=='TAKEN' and first.key==1 and first.payload==b'first'
    assert m.try_publish(2,b'newer')=='PUBLISHED'
    _,state,second=m.poll_once()[0]
    assert second.version==first.version+1 and first.payload==b'first'
    m.slots[0].lock.held=True
    assert m.try_publish(3,b'x')=='BUSY' and m.poll_once()[0][1]=='BUSY'
    assert all(v is False for v in m.slots[0].lock.attempts)
    assert m.try_publish(3,b'x'*17)=='OVERSIZE_OR_MUTABLE'
    assert m.try_publish(3,bytearray(b'x'))=='OVERSIZE_OR_MUTABLE'


def test_committed_ledger_records_are_immutable_and_full_is_explicit():
    ledger=FixedLedger(1);ledger.append(b'complete payload')
    saved=ledger.records()
    with pytest.raises(CapacityFailure):ledger.append(b'overflow')
    assert saved==(b'complete payload',) and ledger.committed==1 and ledger.overflow_record==b'overflow'


def test_copy_time_cannot_backdate_admission():
    p=plant();p.tick();publish(p,1)
    lock=p.results.slots[1].lock
    original_release=lock.release
    def delayed_release():
        original_release()
        p.clock.advance(2)
    lock.release=delayed_release
    p.clock.ns=p.deadline(10)-1
    p.poll_results()
    assert not p.sealed
    event=[e for e in reasons(p) if e['reason']=='RESULT_REJECTED'][-1]
    assert event['rejection']=='LATE' and event['received_ns']==p.deadline(10)+1


def test_bound_configuration_and_immutable_native_limits():
    p=plant()
    with pytest.raises(ValueError):p.limits[0,0]=-999
    with pytest.raises(ValueError):Binding('run','placeholder',digest(b'reference'),0)
    with pytest.raises(ValueError):Binding('run',digest(b'model'),digest(b'reference'),True)
    with pytest.raises(ValueError):plant(max_debt_steps=-1)


def test_unbounded_or_outside_native_command_rejected():
    p=plant();p.tick()
    publish(p,1,command('outside',target=1.0001))
    p.tick()
    assert not p.sealed and any(e['reason']=='RESULT_MALFORMED' for e in reasons(p))
    with pytest.raises(ValueError):
        Command.make('dtype','test',np.zeros(23,np.float32),np.zeros(23,np.float32))


def test_no_ordinary_result_after_latched_miss_can_rearm():
    p=plant(50)
    for _ in range(11):p.tick()
    first=dict(p.input_fault);history=p.history.snapshot();epoch=p.epoch_ns
    publish(p,1,command('too-late',raw=99))
    p.tick()
    assert p.input_fault==first and p.history.snapshot()==history and p.epoch_ns==epoch
    assert not p.sealed and p.active.command_id=='initial'
    assert any(e.get('rejection')=='FAULT_LATCHED' for e in reasons(p))


def test_early_clock_wake_does_not_execute_native_step():
    p=plant()
    p.clock.wait_until_ns=lambda deadline:None
    assert not p.tick()
    assert p.summary()['attempted']==0 and p.summary()['unexecuted']==60
    assert p.failure['reason']=='CLOCK_RETURNED_BEFORE_FIXED_DEADLINE'


def test_unpublished_job_cannot_authenticate_result():
    p=plant();p.jobs.slots[1].lock.held=True;p.tick()
    assert 1 in p.issued and 1 not in p.published_jobs
    publish(p,1);p.tick()
    assert not p.sealed and any(e.get('rejection')=='JOB_NOT_PUBLISHED' for e in reasons(p))


def test_exact_completion_deadline_passes_but_one_ns_late_stays_failed():
    equal=plant(10,stepper={'duration':STEP_NS});equal_summary=run(equal)
    assert equal_summary['timing_passed'] and equal_summary['deadline_misses']==0
    late=plant(10,stepper={'duration':STEP_NS+1});late_summary=run(late)
    assert not late_summary['timing_passed'] and late_summary['deadline_misses']==10
    assert late_summary['first_deadline_failure']['index']==0
    assert late_summary['first_deadline_failure']['lateness_ns']==1


def test_missing_or_duplicated_prepared_clock_ids_fail_before_start():
    with pytest.raises(ValueError):plant(30,window_ids=['one','two'])
    with pytest.raises(ValueError):plant(30,window_ids=['same','same','last'])


def test_logger_cannot_modify_latched_fault_or_timing_records():
    p=plant(50);p.tick();p.clock.advance(23_000_000);run(p)
    original=p.summary()
    with pytest.raises(TypeError):p.first_deadline_failure['index']=999
    with pytest.raises(TypeError):p.input_fault['control']=999
    copy=p.summary();copy['first_deadline_failure']['index']=999;copy['input_fault']['control']=999
    assert p.summary()==original
    failed=plant(50,max_debt_steps=1);failed.tick();failed.clock.advance(23_000_000);failed.tick()
    original_failure=dict(failed.failure)
    with pytest.raises(TypeError):failed.failure['reason']='erased'
    failed.summary()['failure']['reason']='erased'
    assert dict(failed.failure)==original_failure


def test_root_probe_mutable_verifier_issue_is_rejected_and_copied():
    p=plant(1)
    mutable={'message':['original']}
    p.stepper.verify_step=lambda captured:mutable
    assert not p.tick()
    before=p.summary(); evidence=p.last_verifier_return
    mutable['message'][0]='changed through original alias'
    before['failure']['detail']='changed through summary'
    assert p.last_verifier_return==evidence and b'original' in evidence
    assert p.summary()['failure']['detail']!='changed through summary'
    assert (p.returned,p.captured,p.verified,p.step_records.committed)==(1,1,0,0)
    assert p.last_capture is not None and not p.summary()['timing_passed']


def test_root_probe_nested_mutable_warning_has_no_capture_credit_or_alias():
    p=plant(1); original=p.stepper.capture_step; warning={'count':0}
    p.stepper.capture_step=lambda:replace(original(),warnings=(warning,))
    assert not p.tick()
    evidence=p.last_capture_return; warning['count']=99
    assert p.last_capture_return==evidence
    saved_warning=json.loads(evidence)['fields']['warnings']['items'][0][1]
    assert saved_warning['items'][0][1]['value']==0
    assert p.last_capture is None
    assert (p.returned,p.captured,p.verified,p.step_records.committed)==(1,0,0,0)
    assert p.summary()['failure']['stage']=='STEP_CAPTURE'


def test_root_probe_backward_finish_clock_never_passes():
    p=plant(1,stepper={'duration':-1})
    assert not p.tick()
    s=p.summary()
    assert s['failure']['reason']=='CLOCK_NONMONOTONIC_OR_INVALID'
    assert s['clock_fault']['phase']=='STEP_FINISH' and not s['timing_passed']
    assert s['current_debt'] is None
    assert (p.attempted,p.returned,p.captured,p.verified,p.step_records.committed)==(1,1,1,1,0)
    assert p.last_capture.simulation_time==.002


def test_valid_warning_ledger_is_deeply_immutable():
    p=plant(1); run(p)
    snapshot=p.step_records.records()[0].captured
    with pytest.raises(TypeError):snapshot.warnings.counts[0]=99
    with pytest.raises(Exception):snapshot.warnings.lastinfo=(99,)*8
    assert snapshot.warnings==WarningLedger((0,)*8,(0,)*8)


@pytest.mark.parametrize('field,value',[
    ('simulation_time', float('nan')), ('simulation_time',float('inf')), ('simulation_time',-.1),
    ('simulation_time',0), ('state',b''), ('state',bytearray(b'alias')),
    ('torque',b'bad-size'), ('torque',np.full(23,np.nan,np.float64).tobytes()),
    ('warnings',WarningLedger([0]*8,(0,)*8)),
    ('warnings',WarningLedger((False,)+(0,)*7,(0,)*8)),
    ('warnings',WarningLedger((-1,)+(0,)*7,(0,)*8)),
    ('warnings',WarningLedger((0,)*8,(2**31,)+(0,)*7)),
    ('warnings',WarningLedger((0,)*7,(0,)*8))])
def test_complete_capture_schema_before_credit(field,value):
    p=plant(1); original=p.stepper.capture_step
    p.stepper.capture_step=lambda:replace(original(),**{field:value})
    assert not p.tick()
    assert p.returned==1 and p.captured==p.verified==p.step_records.committed==0
    assert p.last_capture is None and type(p.last_capture_return) is bytes


@pytest.mark.parametrize('value',['',False,0,[],{'issue':'mutable'}])
def test_invalid_verifier_contract_does_not_credit_verified(value):
    p=plant(1);p.stepper.verify_step=lambda capture:value
    assert not p.tick()
    assert p.captured==1 and p.verified==p.step_records.committed==0
    assert type(p.last_verifier_return) is bytes


@pytest.mark.parametrize('value',[True,100000000.5,-1])
def test_noninteger_negative_or_bool_wall_clock_fails_before_attempt(value):
    p=plant(1);p.clock.wait_until_ns=lambda deadline:None;p.clock.ns=value
    assert not p.tick()
    assert p.attempted==0 and p.failure['reason']=='CLOCK_NONMONOTONIC_OR_INVALID'
    assert not p.summary()['timing_passed']


def test_backward_clock_between_steps_is_latched_without_second_attempt():
    p=plant(2);assert p.tick()
    p.clock.wait_until_ns=lambda deadline:None;p.clock.ns-=1
    assert not p.tick()
    assert p.attempted==p.returned==1 and not p.summary()['timing_passed']


def test_opaque_invalid_capture_return_is_owned_and_not_called():
    class Invalid:
        def __repr__(self):raise AssertionError('custom repr must not run')
    p=plant(1);p.stepper.capture_step=lambda:Invalid()
    assert not p.tick()
    assert p.captured==0 and p.last_capture is None
    assert b'unsupported' in p.last_capture_return


def test_invalid_clock_in_external_poll_latches_and_never_seals():
    p=plant();assert p.tick();publish(p,1)
    p.clock.ns-=1
    with pytest.raises(RuntimeError):p.poll_results()
    assert not p.sealed and p.failure['reason']=='CLOCK_NONMONOTONIC_OR_INVALID'
    assert not p.tick()


def test_summary_cannot_erase_first_clock_fault_or_publish_aliases():
    p=plant(1,stepper={'duration':-1});p.tick()
    first=dict(p.clock_fault);s=p.summary()
    s['clock_fault']['phase']='erased';s['failure']['phase']='erased'
    p.clock.ns=200_000_000
    assert p.summary()['clock_fault']==first
    assert p.summary()['failure']['phase']=='STEP_FINISH'
    assert not p.summary()['timing_passed']
