"""Accounting-only proxies. No model creation or numerical imports at import time.

Attempt counters advance immediately before the wrapped operation. A cap refusal
does not consume an attempt; a wrapped exception does. Return counters advance
only after a normal return. Units for interrupted multistep Batch calls are
attempted/returned lane-steps, never a claim of completed internal steps.
"""
import copy
import functools
from contextlib import contextmanager
from recovery_contract import budgets


class BudgetExceeded(RuntimeError):
    pass


class WorkLedger:
    def __init__(self, limits=None):
        self.limits = budgets() if limits is None else dict(limits)
        self.counts = {key:dict(attempted_calls=0, returned_calls=0,
                               attempted_units=0, returned_units=0)
                       for key in self.limits}
        self.active = []
        self.first_failure = None
        self.refused = None
        self.phase = 'before_setup'

    def ensure_available(self,key,units):
        if type(units) is not int or units < 0:
            raise ValueError('Invalid work units')
        counts = self.counts[key]
        max_calls, max_units = self.limits[key]
        if counts['attempted_calls']+1 > max_calls or counts['attempted_units']+units > max_units:
            failure = dict(category=key, units=units, phase=self.phase,
                           kind='budget_refusal', counts=counts.copy())
            if self.refused is None:self.refused=failure
            raise BudgetExceeded('Selected work ceiling: '+key)

    def invoke(self, key, units, call, *args, **kwargs):
        self.ensure_available(key,units)
        counts=self.counts[key]
        counts['attempted_calls']+=1
        counts['attempted_units']+=units
        operation = dict(category=key, units=units, ordinal=counts['attempted_calls'], phase=self.phase)
        self.active.append(operation)
        try:
            result=call(*args, **kwargs)
        except BaseException as exc:
            if self.first_failure is None:
                self.first_failure=dict(operation, type=type(exc).__name__, message=str(exc),
                                        stack=copy.deepcopy(self.active))
            raise
        else:
            counts['returned_calls']+=1
            counts['returned_units']+=units
            return result
        finally:
            popped=self.active.pop()
            assert popped is operation

    def snapshot(self):
        return dict(phase=self.phase, counts=copy.deepcopy(self.counts),
            limits={k:dict(attempted_calls_max=v[0],attempted_units_max=v[1]) for k,v in self.limits.items()},
            active=copy.deepcopy(self.active), first_failure=copy.deepcopy(self.first_failure),
            refused=copy.deepcopy(self.refused), model_training_updates=0,
            batch_partial_call_internal_progress_known=False,
            implicit_batch_forward_calls_not_separately_wrapped=True)


class BatchProxy:
    def __init__(self, actual, lanes, ledger):
        self._actual=actual;self._lanes=lanes;self._ledger=ledger
        if lanes not in (2460,9):raise ValueError('Unexpected qualified Batch lane count')

    def __getattr__(self,name):return getattr(self._actual,name)

    def _selected(self,ids):
        if ids is None:return self._lanes
        # Qualified source uses unique integer ID arrays, never boolean masks.
        return len(ids)

    def step(self,ids=None,nstep=1):
        if type(nstep) is not int or nstep<1:raise ValueError('Invalid Batch nstep')
        key='batch_fd_step' if self._lanes==2460 else 'batch_line_step'
        units=self._selected(ids)*nstep
        if ids is None:return self._ledger.invoke(key,units,self._actual.step,nstep=nstep)
        return self._ledger.invoke(key,units,self._actual.step,ids,nstep=nstep)

    def forward(self,ids=None):
        units=self._selected(ids)
        if ids is None:return self._ledger.invoke('batch_explicit_forward',units,self._actual.forward)
        return self._ledger.invoke('batch_explicit_forward',units,self._actual.forward,ids)


class SessionProxy:
    def __init__(self,actual,key,ledger):self._actual=actual;self._key=key;self._ledger=ledger
    def __getattr__(self,name):return getattr(self._actual,name)
    def run(self,names,inputs,*args,**kwargs):
        rows=int(inputs['state'].shape[0])
        if self._key=='BFM_actor' and rows!=1:raise ValueError('Actor must receive one row')
        if self._key=='BFM_backward' and not 1<=rows<=8:raise ValueError('Backward rows outside original goal window')
        return self._ledger.invoke(self._key,rows,self._actual.run,names,inputs,*args,**kwargs)


class Hooks:
    """Install in the isolated diagnostic process, before any planner exists."""
    def __init__(self,ledger,mujoco,core,restoration,driver):
        self.ledger=ledger;self.live_data=None;self._patches=[];self.private_context=None
        def patch(obj,name,value):
            self._patches.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
        native_step=mujoco.mj_step
        @functools.wraps(native_step)
        def step(model,data,*args,**kwargs):
            # Qualified Python sources use one native step per call. Do not
            # silently misaccount a newly added multistep overload.
            if args or kwargs:raise ValueError('Unreviewed native mj_step overload')
            if data is self.live_data:return ledger.invoke('native_live_step',1,native_step,model,data)
            if self.private_context is None:raise ValueError('Unclassified private native step')
            category='native_'+self.private_context+'_step'
            # Refuse before either attempted native-step count advances.
            ledger.ensure_available(category,1)
            return ledger.invoke('native_private_step',1,ledger.invoke,category,1,native_step,model,data)
        patch(mujoco,'mj_step',step)
        factory=core.Batch
        def batch(model,num_sims,*args,**kwargs):
            if type(num_sims) is not int or num_sims not in (2460,9):
                raise ValueError('Unexpected Batch construction')
            actual=ledger.invoke('batch_construct',num_sims,factory,model,num_sims,*args,**kwargs)
            return BatchProxy(actual,num_sims,ledger)
        patch(core,'Batch',batch)
        rollout=core.Planner.rollout
        @functools.wraps(rollout)
        def counted_rollout(planner,*args,**kwargs):
            key='restoration_rollout' if planner.feasibility is None else 'ordinary_rollout'
            return ledger.invoke(key,1,rollout,planner,*args,**kwargs)
        patch(core.Planner,'rollout',counted_rollout)
        linearize=core.Planner.linearize
        @functools.wraps(linearize)
        def counted_linearize(planner,*args,**kwargs):
            return ledger.invoke('linearize',1,linearize,planner,*args,**kwargs)
        patch(core.Planner,'linearize',counted_linearize)
        ilqr=core.ilqr
        @functools.wraps(ilqr)
        def counted_ilqr(planner,*args,**kwargs):
            key='restoration_ilqr' if planner.feasibility is None else 'ordinary_ilqr'
            return ledger.invoke(key,1,ilqr,planner,*args,**kwargs)
        for module in (core,restoration,driver):patch(module,'ilqr',counted_ilqr)
        self.ilqr=counted_ilqr
        for module,name,context,category in (
            (driver,'inspect_native_segment','initial_certificate','initial_native_certificate'),
            (restoration,'inspect_native_segment','restoration_certificate','restoration_native_certificate'),
            (driver,'preview_native_control','preview','imminent_native_preview')):
            original=getattr(module,name)
            def make_wrapper(actual,scope,key):
                @functools.wraps(actual)
                def wrapped(*args,**kwargs):
                    with self.private(scope):return ledger.invoke(key,1,actual,*args,**kwargs)
                return wrapped
            patch(module,name,make_wrapper(original,context,category))

    @contextmanager
    def private(self,context):
        if self.private_context is not None:raise ValueError('Unexpected nested native private context')
        self.private_context=context
        try:yield
        finally:self.private_context=None

    def register_live(self,data):
        if self.live_data is not None:raise ValueError('Actual plant already registered')
        self.live_data=data

    def instrument_seed(self,seed):
        if getattr(seed,'_recovery_instrumented',False):raise ValueError('Seed already instrumented')
        for name in ('actor','backward'):
            seed.sessions[name]=SessionProxy(seed.sessions[name],'BFM_'+name,self.ledger)
        propose=seed.propose
        @functools.wraps(propose)
        def counted(*args,**kwargs):
            with self.private('BFM'):return self.ledger.invoke('BFM_propose',1,propose,*args,**kwargs)
        seed.propose=counted;seed._recovery_instrumented=True

    def close(self):
        for obj,name,value in reversed(self._patches):setattr(obj,name,value)
        self._patches=[]
