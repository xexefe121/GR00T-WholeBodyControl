"""Deterministic clocks/stubs only; no POSIX timer, worker or native execution."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from stage_watchdog import BUDGETS,NS,StageWatchdog,StageTimeout,StageJournal,counters_only,validate_request


def valid_request():
    return dict(watchdog_budgets=dict(BUDGETS),epoch_lead_ns=200000000,debt_abort_steps=100,
        elapsed_abort_ns=60*NS,native_step_budget=18190,serialization_budget=4,epoch_rebase_allowed=False,
        outer_process_timeout_seconds=555)


def timer():
    now=[0];arms=[];return now,arms,StageWatchdog(lambda:now[0],arms.append)


def test_setup_does_not_consume_single_epoch_plant_budget():
    now,arms,w=timer();w.setup();now[0]=127*NS
    epoch=now[0]+200000000;w.plant(epoch)
    assert w.deadline==epoch+120*NS and w.transitions[0]['deadline_ns']==240*NS
    now[0]=epoch+36380000000;w.check();w.preserve()
    assert w.deadline==now[0]+180*NS
    now[0]+=100*NS;w.finish();assert arms[-1]==0 and w.phase=='done'


@pytest.mark.parametrize('stage',['setup','plant','preservation'])
def test_exact_deadline_fails_and_no_stage_rearm(stage):
    now,arms,w=timer();w.setup()
    if stage!='setup':w.plant(200000000)
    if stage=='preservation':w.preserve()
    now[0]=w.deadline
    with pytest.raises(StageTimeout) as error:w.check()
    assert error.value.stage==stage and error.value.observed_ns==w.deadline
    assert not isinstance(error.value,Exception)
    with pytest.raises(ValueError):w.setup()
    if stage!='setup':
        with pytest.raises(ValueError):w.plant(now[0]+200000000)


def test_late_setup_cannot_arm_plant():
    now,arms,w=timer();w.setup();now[0]=240*NS
    with pytest.raises(StageTimeout):w.plant(now[0]+200000000)
    assert w.phase=='setup'


def test_early_signal_keeps_absolute_deadline():
    now,arms,w=timer();w.setup();deadline=w.deadline;now[0]=deadline-1
    w.on_alarm();assert w.deadline==deadline and arms[-1]==1/NS and len(w.transitions)==1


def test_delayed_native_delivery_reports_observed_time_without_inventing_returns():
    now,arms,w=timer();w.setup();w.plant(200000000);deadline=w.deadline;now[0]=deadline+9*NS
    with pytest.raises(StageTimeout):w.on_alarm()
    assert w.timeouts[-1]==dict(stage='plant',deadline_ns=deadline,observed_ns=now[0])
    w.preserve();assert w.deadline==now[0]+180*NS


@pytest.mark.parametrize('value',[True,-1,0.5,float('inf')])
def test_bad_clock_rejected(value):
    now,arms,w=timer();now[0]=value
    with pytest.raises(ValueError):w.setup()


def test_backward_clock_rejected():
    now,arms,w=timer();now[0]=10;w.setup();now[0]=9
    with pytest.raises(ValueError):w.check()


def test_owned_bookkeeping_no_native_reads_or_aliases():
    api=SimpleNamespace(step_attempted=7,step_returned=6,serialization_attempted=2,serialization_returned=2)
    native=SimpleNamespace(attempted=7,returned=6,capture_attempts=6,captured=6,verification_attempts=6,verified=6,stage='NATIVE_STEP_ATTEMPT')
    foundation=SimpleNamespace(attempted=7,returned=6,captured=6,verified=6)
    out=counters_only(SimpleNamespace(foundation=foundation),native,api);api.step_returned=7
    assert out['api']['step_returned']==6 and out['native_return_may_be_uncertain'] is True
    assert out['extra_capture_or_native_read'] is False
    assert counters_only(None,None,None)['api'] is None


def test_durable_journal_once_and_fsync(tmp_path):
    with patch('stage_watchdog.os.fsync') as sync:
        journal=StageJournal(tmp_path/'stages',lambda:4,'a'*64)
        p=journal.record('epoch_armed',{'epoch_ns':20});assert sync.call_count==1
        assert json.loads(p.read_text())['epoch_ns']==20
        journal.record('plant_or_setup_exit',{'counters':{'returned':0}})
        assert len(list((tmp_path/'stages').glob('*')))==2
    with pytest.raises(FileExistsError):StageJournal(tmp_path/'stages',lambda:4,'a'*64)


@pytest.mark.parametrize('key,value',[('debt_abort_steps',101),('elapsed_abort_ns',61*NS),('epoch_rebase_allowed',True),('native_step_budget',18191)])
def test_original_limits_remain_literal(key,value):
    r=valid_request();validate_request(r);r[key]=value
    with pytest.raises(ValueError):validate_request(r)


def test_new_budgets_cannot_be_silently_changed():
    r=valid_request();r['watchdog_budgets']['plant_ns']+=1
    with pytest.raises(ValueError):validate_request(r)


def test_hot_loop_ast_unchanged_and_durable_exit_precedes_cleanup():
    current=Path(__file__).with_name('run_clock.py');prior=current.parents[2]/'independent_plant_process_clock_v1/source_runner_v1/run_clock.py'
    def loop(path):
        tree=ast.parse(path.read_text());return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.While)]
    assert loop(current)==loop(prior)
    text=current.read_text();assert text.index("journal.record('plant_or_setup_exit'")<text.index('cleanup=lifecycle.stop_join()')
    assert text.index("journal.record('epoch_armed'")<text.index('while session.foundation.returned<18190:')


def test_posix_binding_uses_only_injected_stub_and_restores_handler():
    import sys
    from stage_watchdog import PosixWatchdog
    calls=[];fake=SimpleNamespace(ITIMER_REAL=1,SIGALRM=2,
        getitimer=lambda value:(0.,0.),getsignal=lambda value:'old',
        setitimer=lambda *args:calls.append(('timer',args)),signal=lambda *args:calls.append(('handler',args)))
    with patch.dict(sys.modules,{'signal':fake}):
        with PosixWatchdog(lambda:0) as w:w.setup()
    assert calls[1]==('timer',(1,240.))
    assert calls[-2:]==[('timer',(1,0)),('handler',(2,'old'))]


def test_late_epoch_record_aborts_without_tick_or_rebase():
    from run_clock import require_epoch
    now,arms,w=timer();w.setup();epoch=200000000;w.plant(epoch)
    # A slow fsync/allocation is included in the final same-epoch gate.
    now[0]=epoch
    with pytest.raises(RuntimeError):require_epoch(0,epoch,now[0])
    assert w.transitions[1]['deadline_ns']==epoch+120*NS
    assert len([x for x in w.transitions if x['stage']=='plant'])==1
