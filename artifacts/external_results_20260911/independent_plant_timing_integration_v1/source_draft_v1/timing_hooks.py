"""Preallocated scope adapters. No real clocks, callback registry or native imports."""

class InstrumentationFault(RuntimeError):pass

class _NullScope:
    def __enter__(self):return self
    def __exit__(self,kind,value,tb):return False

NULL=_NullScope()

class _Scope:
    __slots__=('owner','phase','tick','control','token','native')
    def __init__(self,owner):self.owner=owner;self.phase=self.tick=self.control=0;self.token=-1;self.native=False
    def __enter__(self):
        h=self.owner
        try:
            if self.native:h.require_before_native()
            try:self.token=h.probe.begin(self.phase,self.tick,self.control)
            except Exception:h.record_fault('begin',self.token)
            except BaseException:
                h.record_fault('fatal_begin',self.token)
                raise
            if self.native:h.require_before_native()
            return self
        except BaseException:
            h.depth-=1
            raise
    def __exit__(self,kind,value,tb):
        h=self.owner
        try:
            if self.token>=0:
                try:
                    h.probe.mark_returned(self.token,raised=kind is not None)
                    h.probe.end(self.token)
                except Exception:h.record_fault('end',self.token)
                except BaseException:
                    # Preserve an already-raised measured exception as first cause.
                    # If there is no measured exception, propagate watchdog/interrupt.
                    h.record_fault('fatal_end',self.token)
                    if kind is None:raise
        finally:h.depth-=1
        return False

class TimingHooks:
    def __init__(self,probe=None):
        self.probe=probe;self.depth=0;self.first_hook_fault=None;self.hook_faults=0
        self.scopes=tuple(_Scope(self) for _ in range(16)) if probe is not None else ()
    @property
    def failed(self):return self.first_hook_fault is not None or (self.probe is not None and self.probe.fault_code!=0)
    def record_fault(self,phase,token):
        self.hook_faults+=1
        if self.first_hook_fault is None:self.first_hook_fault=(phase,token)
    def status(self):
        return dict(enabled=self.probe is not None,failed=self.failed,first_hook_fault=self.first_hook_fault,
            hook_faults=self.hook_faults,active_scopes=self.depth,
            probe_fault_code=None if self.probe is None else self.probe.fault_code,
            spans_started=0 if self.probe is None else self.probe.spans_started,
            spans_closed=0 if self.probe is None else self.probe.spans_closed,
            gc_recorded=0 if self.probe is None else self.probe.gc_recorded)
    def span(self,phase,tick,control):
        if self.probe is None:return NULL
        if self.depth>=len(self.scopes):self.record_fault('scope_capacity',-1);return NULL
        scope=self.scopes[self.depth];self.depth+=1
        scope.phase,scope.tick,scope.control,scope.token,scope.native=phase,tick,control,-1,False
        return scope
    def native_span(self,phase,tick,control):
        scope=self.span(phase,tick,control)
        if scope is not NULL:scope.native=True
        return scope
    def require_before_native(self):
        if self.failed:raise InstrumentationFault('Timing evidence fault; no next native step')
