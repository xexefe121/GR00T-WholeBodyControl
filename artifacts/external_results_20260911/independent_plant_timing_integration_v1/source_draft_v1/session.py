"""Composition of unchanged foundation and injected stepper/mailboxes.

No runtime loader or executable entry point. Stub tests inject every side effect.
"""
from dataclasses import asdict
import numpy as np
from timing_hooks import TimingHooks
from clock_core import PlantFoundation,FixedLedger,encode,digest,STEP_NS,owned_evidence
from shared_mailbox import SharedMailbox,epoch_token


class DeadlineClock:
    def __init__(self,now_ns,sleep):self.read,self.sleep=now_ns,sleep;self.previous=None
    def now_ns(self):
        value=self.read()
        if type(value) is not int or value<0 or (self.previous is not None and value<self.previous):
            raise ValueError('monotonic nonnegative integer clock required')
        self.previous=value;return value
    def wait_until_ns(self,deadline):
        if type(deadline) is not int or deadline<0:raise ValueError('fixed integer deadline')
        while True:
            now=self.now_ns()
            if now>=deadline:return
            self.sleep((deadline-now)/1e9)


class ObservedEndpoint:
    """Log transport-specific statuses without remapping or extra lock attempts."""
    def __init__(self,endpoint,clock,ledger,direction):
        self.endpoint,self.clock,self.ledger,self.direction=endpoint,clock,ledger,direction
    def try_publish(self,key,payload):
        start=self.clock.now_ns();status=self.endpoint.try_publish(key,payload);end=self.clock.now_ns()
        self.ledger.append(encode(dict(direction=self.direction,operation='publish',key=key,status=status,
            start_ns=start,end_ns=end,payload_sha256=digest(payload) if type(payload) is bytes else None)))
        return status
    def poll_once(self):
        start=self.clock.now_ns();items=self.endpoint.poll_once();end=self.clock.now_ns()
        for slot,status,publication in items:
            if status!='EMPTY':
                self.ledger.append(encode(dict(direction=self.direction,operation='poll',slot=slot,status=status,
                    start_ns=start,end_ns=end,key=None if publication is None else publication.key,
                    version=None if publication is None else publication.version,
                    payload_sha256=None if publication is None else digest(publication.payload))))
        return items


def make_channels(context,binding):
    # Future selected setup only; never called by source-only tests.
    token=epoch_token(encode(asdict(binding)))
    return SharedMailbox.create(context,2,32768,token),SharedMailbox.create(context,2,8192,token)


class Session:
    def __init__(self,*,clock,stepper,epoch_ns,binding,table,initial_previous_raw,limits,jobs,results,
                 main_controls=1569,event_capacity=80000,transport_capacity=80000,max_debt_steps=100,max_elapsed_ns=60_000_000_000,timing=None):
        if not 0<main_controls<=len(table.commands):raise ValueError('literal main/hold boundary')
        self.clock,self.stepper,self.table=clock,stepper,table;self.main_controls=main_controls
        self.transport=FixedLedger(transport_capacity);self.outer=FixedLedger(len(table.commands)*10)
        self.foundation=PlantFoundation(clock=clock,stepper=stepper,epoch_ns=epoch_ns,steps=len(table.commands)*10,
            binding=binding,initial_command=table.commands[0],initial_previous_raw=initial_previous_raw,native_limits=limits,
            window_ids=table.windows,jobs=ObservedEndpoint(jobs,clock,self.transport,'plant-jobs'),
            results=ObservedEndpoint(results,clock,self.transport,'plant-results'),event_capacity=event_capacity,
            max_debt_steps=max_debt_steps,max_elapsed_ns=max_elapsed_ns)
        self.timing=TimingHooks() if timing is None else timing
        self.foundation.timing=self.timing
        self.stepper.timing=self.timing
        self.closed=False;self.driver_failure=None
    def tick_once(self):
        with self.timing.span(1,self.foundation.returned,self.foundation.returned//10):
            if self.closed or self.driver_failure is not None:return False
            if self.foundation.returned>=self.foundation.requested_steps:return False
            if self.timing.failed:
                self.driver_failure=encode(dict(type='TimingInstrumentationFailure',detail=self.timing.status(),phase='before-next-tick'))
                return False
            before=self.foundation.returned
            try:
                okay=self.foundation.tick();finish=self.clock.now_ns()
                with self.timing.span(15,before,before//10):
                    self.outer.append(encode(dict(index=before,returned=self.foundation.returned,cycle_return_ns=finish,
                                                  fixed_nominal_end_ns=self.foundation.deadline(before+1))))
                return okay
            except Exception as exc:
                self.driver_failure=encode(dict(type=type(exc).__name__,detail=str(exc),attempted=self.foundation.attempted,
                                               returned=self.foundation.returned,phase='outer-tick-accounting'))
                return False
    def command_accounting(self):
        records=self.foundation.control_records.records();mismatches=[]
        for record in records:
            expected=self.table.commands[record.control]
            if record.command!=expected or record.active_window_id!=self.table.windows[record.control] or record.held:
                mismatches.append(record.control)
        return dict(recorded_controls=len(records),requested_controls=len(self.table.commands),mismatch_controls=mismatches,
                    command_coverage_exact=len(records)==len(self.table.commands) and not mismatches,
                    held_controls=[r.control for r in records if r.held])
    def summary(self):
        core=self.foundation.summary();commands=self.command_accounting()
        native=self.stepper.counters()
        outer=[__import__('json').loads(v) for v in self.outer.records()]
        outer_misses=[v['index'] for v in outer if v['cycle_return_ns']>v['fixed_nominal_end_ns']]
        counters_equal=all(core[k]==native[k] for k in ('attempted','returned','captured','verified'))
        identity_pass=self.closed and native['model_identity']['exit_verified']
        return dict(foundation=core,commands=commands,stepper=native,outer_deadline_miss_indices=outer_misses,
                    application_vs_native_counters_equal=counters_equal,driver_failure=None if self.driver_failure is None else self.driver_failure.decode(),
                    component_preliminary_pass=core['timing_passed'] and core['input_fault'] is None and commands['command_coverage_exact']
                        and counters_equal and not outer_misses and self.driver_failure is None and identity_pass,
                    postrun_model_identity_pass=identity_pass,
                    independent_saved_trace_acceptance_pending=True,component_qualified=False,online_policy_qualified=False,
                    real_time_controller_qualified=False,main_controls=self.main_controls,hold_controls=len(self.table.commands)-self.main_controls)
    def close_native(self):
        if self.closed:raise ValueError('single exit identity verification only')
        self.closed=True
        return self.stepper.verify_model_exit()


class NativeSetupFailure(RuntimeError):
    def __init__(self,evidence):super().__init__('Native setup failed; owned first-failure evidence retained');self.evidence=evidence


def create_native_adapter(model,data,api,contract,expected_mjb,expected_sha256,fixture,counts,lastinfo):
    """Future selected zero-step setup, before admitting the fixed epoch.

    No real model is constructed/imported by this scaffold; caller must use the
    reviewed native loader and literal runtime/input pins. Initialization errors
    require best-effort exit identity verification and preserving adapter failure.
    """
    from native_stepper import NativeStepper
    adapter=None
    try:
        adapter=NativeStepper(model,data,api,contract,expected_mjb,expected_sha256)
        adapter.restore_initial(fixture,counts,lastinfo)
    except Exception as first:
        exit_error=None
        if adapter is not None:
            try:adapter.verify_model_exit()
            except Exception as exc:exit_error=dict(type=type(exc).__name__,detail=str(exc))
        evidence=dict(first_error=dict(type=type(first).__name__,detail=str(first)),exit_error=exit_error,
                      adapter_constructed=adapter is not None,
                      adapter_fault=None if adapter is None or adapter.failure is None else asdict(adapter.failure),
                      counters=None if adapter is None else adapter.counters())
        raise NativeSetupFailure(owned_evidence(evidence)) from first
    return adapter
