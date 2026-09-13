"""Preserve v1 and derive reentrant sampling correction; no real clock calls."""
from pathlib import Path
import hashlib,json,difflib
BASE=Path(__file__).resolve().parent;OLD=BASE/'source_draft_v1';NEW=BASE/'source_draft_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    NEW.mkdir(exist_ok=False)
    for p in OLD.glob('*.py'):
        with (NEW/p.name).open('xb') as f:f.write(p.read_bytes())
    p=NEW/'timing_probe.py';text=p.read_text()
    replacements=[
        ('self.last_wall=self.last_process=-1','self.last_root_wall=self.last_root_process=self.last_root_thread_cpu=-1\n        self.owner_thread=None'),
        ("            if i in (0,3):\n                if value<self.last_wall:return self._fault('CLOCK_REGRESSION',record)\n                self.last_wall=value\n            elif i==2:\n                if value<self.last_process:return self._fault('CLOCK_REGRESSION',record)\n                self.last_process=value\n        return True", "        # A GC callback can complete while a parent reader is in flight.\n        # Compare only this sample's own bracket, never a callback-updated global.\n        if target[offset+3]<target[offset]:return self._fault('CLOCK_REGRESSION',record)\n        return True"),
        ("        if not self._integer(thread,1):self._fault('BAD_ARGUMENT');return -1", "        if not self._integer(thread,1):self._fault('BAD_ARGUMENT');return -1\n        if self.owner_thread is None:self.owner_thread=thread\n        elif thread!=self.owner_thread:self._fault('FOREIGN_SPAN_THREAD');return -1"),
        ("        try:self.spans[offset+14]=int(self._sample(self.spans,offset+6,token))", "        try:\n            okay=self._sample(self.spans,offset+6,token)\n            if okay and parent==-1:\n                if (self.spans[offset+6]<self.last_root_wall or self.spans[offset+7]<self.last_root_thread_cpu\n                    or self.spans[offset+8]<self.last_root_process):okay=self._fault('CLOCK_REGRESSION',token)\n            self.spans[offset+14]=int(okay)"),
        ("                if self.spans[offset+11]<self.spans[offset+7]:okay=self._fault('CLOCK_REGRESSION',token)", "                if (self.spans[offset+10]<self.spans[offset+9] or self.spans[offset+11]<self.spans[offset+7]\n                    or self.spans[offset+12]<self.spans[offset+8]):okay=self._fault('CLOCK_REGRESSION',token)"),
        ("        if okay:self.spans_closed+=1", "        if okay:\n            self.spans_closed+=1\n            if self.spans[offset+4]==-1:\n                self.last_root_wall=self.spans[offset+13]\n                self.last_root_thread_cpu=self.spans[offset+11]\n                self.last_root_process=self.spans[offset+12]"),
        ("                if self.gc[opening+2]!=thread or self.gc[offset+8]<self.gc[opening+8]:self._fault('GC_PAIR',row)", "                if (self.gc[opening+2]!=thread or self.gc[offset+7]<self.gc[opening+10]\n                    or self.gc[offset+8]<self.gc[opening+8] or self.gc[offset+9]<self.gc[opening+9]):self._fault('GC_PAIR',row)"),
    ]
    for old,new in replacements:
        assert text.count(old)==1,old;text=text.replace(old,new)
    p.write_text(text)
    with (BASE/'probe_v1_to_v2.diff').open('x') as f:f.write(''.join(difflib.unified_diff((OLD/p.name).read_text().splitlines(True),text.splitlines(True),fromfile='v1/'+p.name,tofile='v2/'+p.name)))
    with (BASE/'probe_v2_derivation.json').open('x') as f:json.dump(dict(prior_source_sha256={p.name:sha(p) for p in OLD.glob('*.py')},source_only=True,actual_clock_calls=0),f,indent=2)
if __name__=='__main__':main()
