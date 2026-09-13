"""Bounded timing sidecar prototype; injected clocks, no native or GC installation.

Storage is allocated up front. Python clock integers and interpreter execution
still allocate and cost time: this is deliberately not an allocation-free claim.
Expected clock/schema/capacity faults latch numeric evidence. The future integration
must preserve an original measured exception even if an unexpected hook fault occurs.
"""
from array import array
import sys

MAX_SPANS=262144
MAX_GC_EVENTS=4096
PHASES=('TICK_ENVELOPE','RESULT_POLL_AND_ADMIT','FIXED_WAIT','BOUNDARY_SNAPSHOT',
        'BOUNDARY_HISTORY','BOUNDARY_SERIALIZATION','JOB_PUBLICATION','PD_AND_INPUT',
        'MJ_STEP','CAPTURE','CAPTURE_OWNERSHIP','VERIFY','VERIFY_OWNERSHIP','STEP_LEDGER','OUTER_LOG')
SPAN_FIELDS=('phase','tick','control','thread','parent','return_kind',
             'start_wall_before','start_thread_cpu','start_process_cpu','start_wall_after',
             'end_wall_before','end_thread_cpu','end_process_cpu','end_wall_after',
             'begin_complete','end_complete')
GC_FIELDS=('phase','generation','thread','active_span','inside_probe','collected','uncollectable',
           'wall_before','thread_cpu','process_cpu','wall_after','sample_complete')
FAULTS=('NONE','CAPACITY','BAD_ARGUMENT','CLOCK_EXCEPTION','CLOCK_VALUE','CLOCK_REGRESSION',
        'STACK_ORDER','RETURN_MARK','GC_SCHEMA','GC_PAIR','GC_REGISTRY','FOREIGN_SPAN_THREAD')
I64_MAX=(1<<63)-1

class TimingProbe:
    def __init__(self,*,wall_ns,thread_cpu_ns,process_cpu_ns,thread_id,span_capacity=MAX_SPANS,gc_capacity=MAX_GC_EVENTS):
        if type(span_capacity) is not int or not 0<span_capacity<=MAX_SPANS:raise ValueError('span capacity')
        if type(gc_capacity) is not int or not 0<gc_capacity<=MAX_GC_EVENTS:raise ValueError('GC capacity')
        self.readers=(wall_ns,thread_cpu_ns,process_cpu_ns,wall_ns);self.thread_id=thread_id
        self.span_capacity,self.gc_capacity=span_capacity,gc_capacity
        self.spans=array('q',[-1])*(span_capacity*len(SPAN_FIELDS))
        self.gc=array('q',[-1])*(gc_capacity*len(GC_FIELDS))
        self.stack=array('q',[-1])*16;self.depth=0
        self.gc_open=array('q',[-1])*3
        self.spans_attempted=self.spans_started=self.spans_closed=self.spans_denied=0
        self.gc_attempted=self.gc_recorded=self.gc_denied=0
        self.last_root_wall=self.last_root_process=self.last_root_thread_cpu=-1
        self.owner_thread=None
        self.fault_code=0;self.fault_record=-1;self.in_probe=0
        self.registry=None;self.callback=self.gc_event

    def _fault(self,name,record=-1):
        if self.fault_code==0:self.fault_code=FAULTS.index(name);self.fault_record=record
        return False

    @staticmethod
    def _integer(value,minimum=0):return type(value) is int and minimum<=value<=I64_MAX

    def _sample(self,target,offset,record):
        for i,reader in enumerate(self.readers):
            try:value=reader()
            except Exception:return self._fault('CLOCK_EXCEPTION',record)
            if not self._integer(value):return self._fault('CLOCK_VALUE',record)
            target[offset+i]=value
        # A GC callback can complete while a parent reader is in flight.
        # Compare only this sample's own bracket, never a callback-updated global.
        if target[offset+3]<target[offset]:return self._fault('CLOCK_REGRESSION',record)
        return True

    def begin(self,phase,tick,control):
        self.spans_attempted+=1
        if not (self._integer(phase,1) and phase<=len(PHASES) and self._integer(tick) and self._integer(control)):
            self._fault('BAD_ARGUMENT');return -1
        if self.spans_started>=self.span_capacity or self.depth==len(self.stack):
            self.spans_denied+=1;self._fault('CAPACITY');return -1
        try:thread=self.thread_id()
        except Exception:self._fault('CLOCK_EXCEPTION');return -1
        if not self._integer(thread,1):self._fault('BAD_ARGUMENT');return -1
        if self.owner_thread is None:self.owner_thread=thread
        elif thread!=self.owner_thread:self._fault('FOREIGN_SPAN_THREAD');return -1
        parent=int(self.stack[self.depth-1]) if self.depth else -1
        if parent>=0 and self.spans[parent*len(SPAN_FIELDS)+3]!=thread:
            self._fault('FOREIGN_SPAN_THREAD',parent);return -1
        token=self.spans_started;self.spans_started+=1;offset=token*len(SPAN_FIELDS)
        self.spans[offset]=phase;self.spans[offset+1]=tick;self.spans[offset+2]=control
        self.spans[offset+3]=thread;self.spans[offset+4]=parent;self.spans[offset+5]=0
        self.stack[self.depth]=token;self.depth+=1
        prior=self.in_probe;self.in_probe=1
        try:
            okay=self._sample(self.spans,offset+6,token)
            if okay and parent==-1:
                if (self.spans[offset+6]<self.last_root_wall or self.spans[offset+7]<self.last_root_thread_cpu
                    or self.spans[offset+8]<self.last_root_process):okay=self._fault('CLOCK_REGRESSION',token)
            self.spans[offset+14]=int(okay)
        finally:self.in_probe=prior
        return token

    def mark_returned(self,token,*,raised=False):
        # Mark immediately after the measured call returns/raises, before end clocks.
        if not self._integer(token) or token>=self.spans_started or type(raised) is not bool:
            return self._fault('BAD_ARGUMENT',token if type(token) is int else -1)
        offset=token*len(SPAN_FIELDS)
        if not self.depth or self.stack[self.depth-1]!=token:return self._fault('STACK_ORDER',token)
        try:thread=self.thread_id()
        except Exception:return self._fault('CLOCK_EXCEPTION',token)
        if thread!=self.spans[offset+3] or type(thread) is not int:return self._fault('FOREIGN_SPAN_THREAD',token)
        if self.spans[offset+5]!=0:return self._fault('RETURN_MARK',token)
        self.spans[offset+5]=2 if raised else 1
        return True

    def end(self,token):
        if not self._integer(token) or not self.depth or self.stack[self.depth-1]!=token:
            return self._fault('STACK_ORDER',token if type(token) is int else -1)
        offset=token*len(SPAN_FIELDS)
        if self.spans[offset+5]==0:return self._fault('RETURN_MARK',token)
        try:thread=self.thread_id()
        except Exception:return self._fault('CLOCK_EXCEPTION',token)
        if thread!=self.spans[offset+3] or type(thread) is not int:return self._fault('FOREIGN_SPAN_THREAD',token)
        prior=self.in_probe;self.in_probe=1
        try:
            okay=self._sample(self.spans,offset+10,token)
            if okay and self.spans[offset+14]==1:
                if (self.spans[offset+10]<self.spans[offset+9] or self.spans[offset+11]<self.spans[offset+7]
                    or self.spans[offset+12]<self.spans[offset+8]):okay=self._fault('CLOCK_REGRESSION',token)
            self.spans[offset+15]=int(okay)
        finally:self.in_probe=prior
        self.depth-=1;self.stack[self.depth]=-1
        if okay:
            self.spans_closed+=1
            if self.spans[offset+4]==-1:
                self.last_root_wall=self.spans[offset+13]
                self.last_root_thread_cpu=self.spans[offset+11]
                self.last_root_process=self.spans[offset+12]
        return okay

    def gc_event(self,phase,info):
        # Future attach uses exactly this callback; tests inject a plain list.
        self.gc_attempted+=1
        if self.gc_recorded>=self.gc_capacity:
            self.gc_denied+=1;self._fault('CAPACITY');return
        row=self.gc_recorded;self.gc_recorded+=1;offset=row*len(GC_FIELDS)
        try:
            generation=info.get('generation');thread=self.thread_id()
            code=1 if phase=='start' else (2 if phase=='stop' else 0)
            if code==0 or type(info) is not dict or not self._integer(generation) or generation>2 or not self._integer(thread,1):
                self._fault('GC_SCHEMA',row);return
            collected=info.get('collected',0);uncollectable=info.get('uncollectable',0)
            if not self._integer(collected) or not self._integer(uncollectable):self._fault('GC_SCHEMA',row);return
            active=int(self.stack[self.depth-1]) if self.depth else -1
            if active>=0 and self.spans[active*len(SPAN_FIELDS)+3]!=thread:active=-1
            self.gc[offset]=code;self.gc[offset+1]=generation;self.gc[offset+2]=thread
            self.gc[offset+3]=active;self.gc[offset+4]=self.in_probe
            self.gc[offset+5]=collected;self.gc[offset+6]=uncollectable
            self.gc[offset+11]=int(self._sample(self.gc,offset+7,row))
            if code==1:
                if self.gc_open[generation]!=-1:self._fault('GC_PAIR',row)
                else:self.gc_open[generation]=row
            elif self.gc_open[generation]==-1:self._fault('GC_PAIR',row)
            else:
                opening=int(self.gc_open[generation])*len(GC_FIELDS)
                if (self.gc[opening+2]!=thread or self.gc[offset+7]<self.gc[opening+10]
                    or self.gc[offset+8]<self.gc[opening+8] or self.gc[offset+9]<self.gc[opening+9]):self._fault('GC_PAIR',row)
                self.gc_open[generation]=-1
        except Exception:self._fault('GC_SCHEMA',row)

    def attach(self,registry):
        if type(registry) is not list or self.registry is not None:raise ValueError('fresh callback registry required')
        if any(v is self.callback for v in registry):raise ValueError('callback already attached')
        registry.append(self.callback);self.registry=registry

    def detach(self):
        if self.registry is None:return self._fault('GC_REGISTRY')
        matches=[i for i,v in enumerate(self.registry) if v is self.callback]
        if len(matches)!=1:return self._fault('GC_REGISTRY')
        del self.registry[matches[0]];self.registry=None
        return True

    def snapshot(self):
        # Copy only during failure preservation/postrun, never in a 2ms loop.
        return dict(schema_version=1,integer_dtype=self.spans.typecode,integer_itemsize=self.spans.itemsize,integer_byteorder=sys.byteorder,
            span_fields=SPAN_FIELDS,gc_fields=GC_FIELDS,phases=PHASES,
            spans=self.spans[:self.spans_started*len(SPAN_FIELDS)].tobytes(),
            gc=self.gc[:self.gc_recorded*len(GC_FIELDS)].tobytes(),
            spans_attempted=self.spans_attempted,spans_started=self.spans_started,spans_closed=self.spans_closed,
            spans_denied=self.spans_denied,gc_attempted=self.gc_attempted,gc_recorded=self.gc_recorded,gc_denied=self.gc_denied,
            active_depth=self.depth,open_gc_generations=tuple(int(v) for v in self.gc_open),
            fault=FAULTS[self.fault_code],fault_record=self.fault_record,callback_detached=self.registry is None,
            instrumentation_complete=self.fault_code==0 and self.depth==0 and self.spans_started==self.spans_closed
                and all(v==-1 for v in self.gc_open) and self.registry is None,
            no_native_or_timing_qualification=True)
