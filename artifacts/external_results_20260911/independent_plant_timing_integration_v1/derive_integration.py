"""Preserve all producer files and create source-only timing integration."""
from pathlib import Path
import ast,hashlib,json,difflib,textwrap
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
PRIOR=NEW/'independent_plant_pending_result_v1/source_draft_v1'
PROBE=NEW/'independent_plant_timing_instrumentation_v1/source_draft_v2'
SOURCE=BASE/'source_draft_v1';ORIGINAL=BASE/'source_original_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def once(text,a,b):
    assert text.count(a)==1,(a,text.count(a));return text.replace(a,b)
def wrap(text,fragment,header):
    indent=fragment[:len(fragment)-len(fragment.lstrip())]
    replacement=indent+header+'\n'+textwrap.indent(fragment,'    ')
    return once(text,fragment,replacement)
def method(text,cls,name,header):
    tree=ast.parse(text);node=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=text.splitlines(True);fragment=''.join(lines[node.body[0].lineno-1:node.end_lineno])
    return wrap(text,fragment,header)
def main():
    SOURCE.mkdir(exist_ok=False);ORIGINAL.mkdir(exist_ok=False)
    for p in PRIOR.glob('*.py'):
        for folder in (SOURCE,ORIGINAL):
            with (folder/p.name).open('xb') as f:f.write(p.read_bytes())
    for p in PROBE.glob('*.py'):
        with (SOURCE/p.name).open('xb') as f:f.write(p.read_bytes())
    texts={name:(SOURCE/name).read_text() for name in ('clock_core.py','native_stepper.py','session.py','run_clock.py')}
    t=texts['clock_core.py'];t=once(t,'import base64\n','import base64\nfrom timing_hooks import TimingHooks\n')
    t=once(t,'        self.clock_fault = None\n','        self.timing = TimingHooks()\n        self.clock_fault = None\n')
    t=method(t,'PlantFoundation','_attempt_job_publication','with self.timing.span(7,self.returned,self.returned//CONTROL_STEPS):')
    t=wrap(t,'        measured_state, measured_terms = self.stepper.boundary_snapshot()\n','with self.timing.span(4,self.returned,control):')
    a=t.index('        before, flat, terms, after = self.history.advance(');b=t.index('        if control + 1 <',a)
    t=wrap(t,t[a:b],'with self.timing.span(5,self.returned,control):')
    a=t.index('            payload = encode(dict(state=');b=t.index('            self._attempt_job_publication(pending)',a)
    t=wrap(t,t[a:b],'with self.timing.span(6,self.returned,control):')
    t=wrap(t,'            self.poll_results()  # Opportunistic, outside the sealed activation switch.\n','with self.timing.span(2,self.returned,self.returned//CONTROL_STEPS):')
    t=wrap(t,'            self.clock.wait_until_ns(self.deadline(index))\n','with self.timing.span(3,index,index//CONTROL_STEPS):')
    t=once(t,"            self.stage = 'STEP_ATTEMPT'\n","            self.timing.require_before_native()\n            self.stage = 'STEP_ATTEMPT'\n")
    t=wrap(t,'            snapshot = self.stepper.capture_step()\n','with self.timing.span(10,index,index//CONTROL_STEPS):')
    a=t.index('            self.last_capture_return = owned_evidence(snapshot)');b=t.index("            self.stage = 'STEP_VERIFY'",a)
    t=wrap(t,t[a:b],'with self.timing.span(11,index,index//CONTROL_STEPS):')
    t=wrap(t,'            issue = self.stepper.verify_step(snapshot)\n','with self.timing.span(12,index,index//CONTROL_STEPS):')
    a=t.index('            self.last_verifier_return = owned_evidence(issue)');b=t.index("            finish = self._now_ns('STEP_FINISH')",a)
    t=wrap(t,t[a:b],'with self.timing.span(13,index,index//CONTROL_STEPS):')
    a=t.index('            self.step_records.append(StepRecord(');b=t.index('            if issue is not None:',a)
    t=wrap(t,t[a:b],'with self.timing.span(14,index,index//CONTROL_STEPS):')
    texts['clock_core.py']=t
    t=texts['native_stepper.py'];t=once(t,'import numpy as np\n','import numpy as np\nfrom timing_hooks import TimingHooks\n')
    t=once(t,'        self.model,self.data,self.api=model,data,api\n','        self.timing=TimingHooks()\n        self.model,self.data,self.api=model,data,api\n')
    a=t.index("            target=typed_vector(target,23,'native target'");b=t.index("            self.stage='NATIVE_STEP_ATTEMPT'",a)
    t=wrap(t,t[a:b],'with self.timing.span(8,self.returned,self.returned//10):')
    a=t.index("            self.stage='NATIVE_STEP_ATTEMPT'");b=t.index('        except Exception as exc:',a)
    fragment=t[a:b]
    t=wrap(t,fragment,'with self.timing.native_span(9,self.returned,self.returned//10):')
    texts['native_stepper.py']=t
    t=texts['session.py'];t=once(t,'import numpy as np\n','import numpy as np\nfrom timing_hooks import TimingHooks\n')
    t=once(t,'max_elapsed_ns=60_000_000_000):','max_elapsed_ns=60_000_000_000,timing=None):')
    t=once(t,'        self.closed=False;self.driver_failure=None\n','        self.timing=TimingHooks() if timing is None else timing\n        self.foundation.timing=self.timing\n        self.stepper.timing=self.timing\n        self.closed=False;self.driver_failure=None\n')
    t=once(t,'        before=self.foundation.returned\n',"        if self.timing.failed:\n            self.driver_failure=encode(dict(type='TimingInstrumentationFailure',detail=self.timing.status(),phase='before-next-tick'))\n            return False\n        before=self.foundation.returned\n")
    a=t.index('            self.outer.append(encode(dict(index=');b=t.index('            return okay',a)
    t=wrap(t,t[a:b],'with self.timing.span(15,before,before//10):')
    t=method(t,'Session','tick_once','with self.timing.span(1,self.foundation.returned,self.foundation.returned//10):')
    texts['session.py']=t
    # Supervisor wiring is a separate reviewed additive text block.
    for name,t in texts.items():
        compile(t,str(SOURCE/name),'exec');(SOURCE/name).write_text(t)
    diff=''
    for name in texts:diff+=''.join(difflib.unified_diff((ORIGINAL/name).read_text().splitlines(True),texts[name].splitlines(True),fromfile='original/'+name,tofile='integrated/'+name))
    with (BASE/'initial_core_integration.diff').open('x') as f:f.write(diff)
    with (BASE/'original_source_hashes.json').open('x') as f:json.dump({p.name:sha(p) for p in ORIGINAL.glob('*.py')},f,indent=2)
if __name__=='__main__':main()
