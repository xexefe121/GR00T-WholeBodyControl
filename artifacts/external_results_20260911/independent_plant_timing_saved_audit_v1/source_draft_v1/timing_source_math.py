"""Literal instrumented source lineage; pure files/AST, no runtime imports."""
import hashlib,json
import clock_stage_math as original

PREPARATION='3844f7d86a4f97557c8a74744f7ea382a25315ae3a96cb3abfa786992d2c8cc5'
REVIEW='fdccbf0e31e716190a805cd2d5e6f52d3aae629e18136386f4da9d1e95b78310'
CONTRACT='preallocated_wall_thread_process_GC_v2'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def source_contract(roles,pinned):
    prep=roles['timing_source_preparation'];review=roles['timing_source_review']
    if digest(prep)!=PREPARATION or digest(review)!=REVIEW:raise AssertionError('Actual timing source qualification changed')
    p=json.loads(prep.read_text());r=json.loads(review.read_text())
    if r['passed'] is not True or r['source_preparation_sha256']!=PREPARATION or r['source_sha256']!=p['source_sha256']:raise AssertionError('Timing source review subjects differ')
    source=roles['clock_runner'].parent
    current={path.name:digest(path) for path in source.glob('*.py')}
    if current!=p['source_sha256']:raise AssertionError('Instrumented source namespace differs')
    for name,sha in current.items():
        if pinned.get((source/name).resolve())!=sha:raise AssertionError('Actually imported timing source not frozen')
    for role,name in [('clock_core','clock_core.py'),('clock_runner','run_clock.py'),('clock_watchdog','stage_watchdog.py'),
        ('clock_worker','dummy_worker.py'),('clock_protocol','recorded_protocol.py'),('clock_pending_result','pending_result.py')]:
        if roles[role]!=source/name:raise AssertionError('Current clock role namespace differs')
    prior=original.source_contract({key:roles['prior_'+key] for key in original.SOURCE_SHA})
    return dict(prior=prior,literal_instrumented_source_exact=True,review_sha256=REVIEW,preparation_sha256=PREPARATION,
        producer_imports=0,all32_source_pins_exact=True)

def request_contract(request):
    if request.get('timing_probe_contract')!=CONTRACT:raise AssertionError('Explicit reviewed timing probe contract required')
