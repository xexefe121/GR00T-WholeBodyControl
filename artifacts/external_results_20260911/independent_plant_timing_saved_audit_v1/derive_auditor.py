"""Narrow saved-audit adaptation; no actual audit/request or task-array load."""
from pathlib import Path
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'source_draft_v1';OLD=BASE/'source_original_v2'
def once(t,a,b):
    assert t.count(a)==1,(a,t.count(a));return t.replace(a,b)

t=(OLD/'audit_clock.py').read_text()
t=once(t,'import clock_stage_math as stage_math','import clock_stage_math as stage_math\nimport timing_source_math\nimport timing_sidecar_io')
t=once(t,"    stage_source=stage_math.source_contract({key:roles[key] for key in stage_math.SOURCE_SHA})",
    "    timing_source_math.request_contract(run_request)\n    stage_source=timing_source_math.source_contract(roles,pinned)")
t=once(t,"        return dict(stage_watchdog=stage_result,stage_sources=stage_source,evidence_integrity_passed=True,physical_pass=False",
    "        timing_sidecar,_=timing_sidecar_io.read_sidecars(output,report,stage_records,{},pinned)\n        return dict(timing_sidecar=timing_sidecar,stage_watchdog=stage_result,stage_sources=stage_source,evidence_integrity_passed=True,physical_pass=False")
t=once(t,"    counter_result=accounting.counts(a,metadata,report,capsules)",
    "    timing_sidecar,timing_owned=timing_sidecar_io.read_sidecars(output,report,stage_records,a,pinned)\n    counter_result=accounting.counts(a,metadata,report,capsules,timing_sidecar['counter_handoff'])")
t=once(t,'    reserved_capture=accounting.uncommitted_capture(a,metadata,report,capsules,initial[:291],contract)',
    '    reserved_capture=timing_sidecar_io.reserved_capture(a,metadata,report,capsules,initial[:291],contract,timing_owned)')
t=once(t,"    timing_pass=timing_result['fixed_epoch_timing_pass'] and setup_timely and stage_result['stage_watchdog_qualified']",
    "    timing_pass=timing_result['fixed_epoch_timing_pass'] and setup_timely and stage_result['stage_watchdog_qualified'] and timing_sidecar['instrumentation_complete']")
t=once(t,"    return dict(evidence_integrity_passed=True,physical_pass=physical_pass,timing_pass=timing_pass,command_pass=command_pass,",
    "    return dict(timing_sidecar=timing_sidecar,evidence_integrity_passed=True,physical_pass=physical_pass,timing_pass=timing_pass,command_pass=command_pass,")
(SOURCE/'audit_clock.py').write_text(t)

t=(OLD/'clock_accounting.py').read_text()
t=once(t,'def counts(a,metadata,report,capsules):','def counts(a,metadata,report,capsules,timing_handoff=None):')
t=once(t,"    if api['step_attempted']!=native['attempted'] or api['step_returned']!=native['returned']:raise AssertionError('Actual native API counters differ')\n    if native['returned']!=core['returned'] or native['captured']!=core['captured'] or native['verified']!=core['verified']:\n        raise AssertionError('Foundation/native result credit differs')",
    "    api_equal=api['step_attempted']==native['attempted'] and api['step_returned']==native['returned']\n    owner_equal=all(native[k]==core[k] for k in ('returned','captured','verified'))\n    if not (api_equal and owner_equal):\n        if timing_handoff is None or timing_handoff.get('partial_counter_proof') is not True:\n            raise AssertionError('Counter difference lacks independently checked terminal timing handoff')")
t=once(t,"    for _ in range(core['returned']):expected+=.002", "    for _ in range(native['returned']):expected+=.002")
t=once(t,"    return {'foundation':{k:core[k] for k in ('attempted','returned','captured','verified','committed','unexecuted')},",
    "    return {'timing_counter_handoff':timing_handoff,'foundation':{k:core[k] for k in ('attempted','returned','captured','verified','committed','unexecuted')},")
(SOURCE/'clock_accounting.py').write_text(t)

t=(OLD/'prepare_request.py').read_text()
needle="    roles['original_runner']=source.parents[1]/'independent_plant_process_clock_v1/source_runner_v1/run_clock.py'"
t=once(t,needle,needle+"\n    import clock_stage_math\n    prior=source.parents[1]/'independent_plant_pending_result_v1/source_draft_v1'\n    for key in clock_stage_math.SOURCE_SHA:\n        roles['prior_'+key]=roles[key] if key.startswith('original_') else prior/roles[key].name\n    roles['timing_source_preparation']=source.parents[1]/'independent_plant_timing_integration_v1/source_preparation.json'\n    roles['timing_source_review']=source.parents[1]/'independent_timing_integration_root_review_v1/review.json'")
(SOURCE/'prepare_request.py').write_text(t)
for p in SOURCE.glob('*.py'):compile(p.read_text(),str(p),'exec')
print('Derived three modules; preserved all20 originals; no execution.')
