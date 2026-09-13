"""Bound saved JSON envelopes only; no array, model, mailbox or native calls."""
import base64,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
RUN=BASE.parent/'independent_plant_pending_publication_v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def decode(values):
    result=[]
    for row in values:
        assert row['type']=='bytes'
        raw=base64.b64decode(row['base64'],validate=True)
        assert len(raw)==row['length'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        result.append(json.loads(raw))
    return result
report=read(RUN/'run/report.json');owner=read(RUN/'owner_completion.json')
assert sha(RUN/'run/report.json')==owner['report_sha256']=='df8d562e18508b24372b81a7ecab6e8e0d53e195562078dc23ff6a45d54a89f6'
assert sha(RUN/'owner_completion.json')=='a51d7f88bff439a23c696ad48fa6cff5c8ba035cae58820cd1ac2c7f4ccf8b62'
assert owner['accounting_passed'] and owner['process_absence_proven']
for name in ('evidence.json','worker.json'):
    p=RUN/'run'/name;assert owner['output_hashes'][p.as_posix()]==sha(p)
evidence=read(RUN/'run/evidence.json');worker=read(RUN/'run/worker.json')
plant=decode(evidence['transport_records']);foundation=decode(evidence['foundation_events']);outer=decode(evidence['outer_cycles'])
busy=[r for r in plant if r['direction']=='plant-jobs' and r['status']=='BUSY']
busy_keys=sorted({r['key'] for r in busy})
focus=set(busy_keys+[654,655,656,657])
timeline=[]
for origin,rows in [('plant',plant),('worker',worker['events']),('foundation',foundation)]:
    for r in rows:
        if r.get('key',r.get('activation',r.get('control',-1))) in focus:
            timeline.append(dict(origin=origin,**{k:v for k,v in r.items() if k not in ('payload','snapshot_payload')}))
result=dict(report_sha256=sha(RUN/'run/report.json'),owner_sha256=sha(RUN/'owner_completion.json'),
    source_sha256=sha(__file__),verified_envelopes=dict(plant=len(plant),foundation=len(foundation),outer=len(outer)),
    epoch_ns=report['epoch_ns'],busy_keys=busy_keys,busy_count=len(busy),focused_timeline=timeline,
    command656_outer_cycles=[r for r in outer if 6548<=r['index']<=6561],
    source_bound_inspection_only=True,independent_saved_audit_pending=True,model_calls=0,native_steps=0)
with (BASE/'inspection.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(verified_envelopes=result['verified_envelopes'],busy_keys=busy_keys,busy_count=len(busy),
    command656_timeline=[r for r in timeline if r.get('key',r.get('activation',r.get('control',-1)))==656],
    command656_outer_cycles=result['command656_outer_cycles'])))
