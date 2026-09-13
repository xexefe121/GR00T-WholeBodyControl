"""Bounded outer stages, no tick logic. Real timers are installed only by main.

POSIX alarms are a graceful stop request; native C may delay delivery. A separate
outer GNU timeout remains the final hard guard. No handler reads native state.
"""
import json
import os
from pathlib import Path

NS=1_000_000_000
BUDGETS={'setup_ns':240*NS,'plant_ns':120*NS,'preservation_ns':180*NS,
         'outer_timeout_seconds':555,'outer_kill_grace_seconds':5}


def validate_request(request):
    if request.get('watchdog_budgets')!=BUDGETS:
        raise ValueError('literal separately reviewed stage budgets required')
    if any(type(v) is not int for v in request['watchdog_budgets'].values()):
        raise ValueError('integer watchdog budgets required')
    for key,value in {'epoch_lead_ns':200_000_000,'debt_abort_steps':100,
        'elapsed_abort_ns':60*NS,'native_step_budget':18190,'serialization_budget':4,
        'outer_process_timeout_seconds':555}.items():
        if type(request.get(key)) is not int or request[key]!=value:raise ValueError('original strict budget changed: '+key)
    if request.get('epoch_rebase_allowed') is not False:raise ValueError('epoch rebasing forbidden')


class StageTimeout(BaseException):
    """Bypass ordinary native/controller Exception handling; retain uncertainty."""
    def __init__(self,stage,deadline_ns,observed_ns):
        self.stage,self.deadline_ns,self.observed_ns=stage,deadline_ns,observed_ns
        super().__init__(stage+' watchdog deadline reached')


class StageWatchdog:
    def __init__(self,now_ns,arm_seconds):
        self.now=now_ns;self.arm=arm_seconds;self.phase=None;self.deadline=None
        self.previous=None;self.transitions=[];self.timeouts=[]
    def clock(self):
        now=self.now()
        if type(now) is not int or now<0 or (self.previous is not None and now<self.previous):
            raise ValueError('monotonic nonnegative integer watchdog clock required')
        self.previous=now;return now
    def _set(self,phase,now,deadline):
        self.phase=phase;self.deadline=deadline
        self.transitions.append({'stage':phase,'entered_ns':now,'deadline_ns':deadline})
        self.arm((deadline-now)/NS)
    def setup(self):
        if self.phase is not None:raise ValueError('setup may start only once')
        now=self.clock();self._set('setup',now,now+BUDGETS['setup_ns'])
    def plant(self,epoch_ns):
        if self.phase!='setup':raise ValueError('single plant transition required')
        self.check();now=self.clock()
        if type(epoch_ns) is not int or not now<epoch_ns:raise ValueError('plant must arm before the fixed epoch')
        self._set('plant',now,epoch_ns+BUDGETS['plant_ns'])
    def preserve(self):
        if self.phase not in ('setup','plant'):raise ValueError('preservation may start only once')
        now=self.clock();self._set('preservation',now,now+BUDGETS['preservation_ns'])
    def check(self):
        now=self.clock()
        if self.deadline is None:raise ValueError('watchdog is not armed')
        if now>=self.deadline:
            record={'stage':self.phase,'deadline_ns':self.deadline,'observed_ns':now}
            self.timeouts.append(record);raise StageTimeout(**record)
        return now
    def on_alarm(self,*unused):
        # POSIX rounding may deliver early: rearm only the same absolute deadline.
        now=self.check();self.arm((self.deadline-now)/NS)
    def finish(self):
        if self.phase!='preservation':raise ValueError('completion requires preservation phase')
        now=self.check();self.arm(0);self.transitions.append({'stage':'done','entered_ns':now,'deadline_ns':self.deadline});self.phase='done'
    def evidence(self):
        return {'phase':self.phase,'deadline_ns':self.deadline,'transitions':[dict(v) for v in self.transitions],
            'timeouts':[dict(v) for v in self.timeouts],'budgets':dict(BUDGETS),'native_signal_delivery_may_be_delayed':True}


class PosixWatchdog:
    def __init__(self,now_ns):self.now=now_ns
    def __enter__(self):
        import signal
        self.signal=signal
        if signal.getitimer(signal.ITIMER_REAL)!=(0.,0.):raise ValueError('existing interval timer must not be replaced')
        self.old=signal.getsignal(signal.SIGALRM)
        self.watchdog=StageWatchdog(self.now,lambda seconds:signal.setitimer(signal.ITIMER_REAL,seconds))
        signal.signal(signal.SIGALRM,self.watchdog.on_alarm);return self.watchdog
    def __exit__(self,*unused):
        self.signal.setitimer(self.signal.ITIMER_REAL,0)
        self.signal.signal(self.signal.SIGALRM,self.old)


class StageJournal:
    """Small fsynced CreateNew records, called only outside the stepping loop."""
    def __init__(self,folder,now_ns,request_sha):
        self.folder=Path(folder);self.folder.mkdir(exist_ok=False)
        self.now=now_ns;self.request_sha=request_sha;self.index=0
    def record(self,name,payload):
        value={'stage':name,'record_created_ns':self.now(),'request_sha256':self.request_sha,'pid':os.getpid(),**payload}
        path=self.folder/(str(self.index).zfill(2)+'_'+name+'.json')
        # Serialize before opening so invalid data cannot leave a false final record.
        raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        self.index+=1;return path


def counters_only(session,adapter,api):
    """Read bookkeeping attributes only; no capture, native read, step or replay."""
    def counts(value,names):
        if value is None:return None
        return {name:getattr(value,name,None) for name in names}
    foundation=None if session is None else session.foundation
    return {'foundation':counts(foundation,('attempted','returned','captured','verified')),
        'adapter':counts(adapter,('attempted','returned','capture_attempts','captured','verification_attempts','verified','stage')),
        'api':counts(api,('step_attempted','step_returned','serialization_attempted','serialization_returned')),
        'native_return_may_be_uncertain':bool(api is not None and api.step_attempted!=api.step_returned),
        'extra_capture_or_native_read':False}
