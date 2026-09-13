"""Synthetic exclusive-write artifacts and reserved states only; no task data."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from test_timing_saved_math import fixture,raw
from test_clock_accounting import count_case
from timing_sidecar_io import read_sidecars,reserved_capture
from clock_accounting import counts
import timing_saved_math as m
import timing_source_math as source
import clock_stage_math as old_stage

def write_case(path,f):
    blobs={'timing_spans.bin':raw(f['rows'],m.SPAN_FIELDS),'timing_gc.bin':raw(f['gc'],m.GC_FIELDS),
        'timing_metadata.json':json.dumps(f['metadata']).encode(),'timing_owned_partial.json':b'{}'}
    pins={};files={}
    for name,data in blobs.items():
        p=path/name;p.write_bytes(data);sha=hashlib.sha256(data).hexdigest()
        pins[p.resolve()]=sha;files[name]={'sha256':sha,'bytes':len(data)}
    f['report']['timing_instrumentation'].update(files=files,new_native_reads=0)
    return pins

def reserved():
    a,metadata,report,capsules,initial,contract,*_=count_case(2)
    state=a['packed_capture'][-1].copy();state[0]+=.002;command=a['step_command'][-1].copy()
    value={'dataclass':'CapturedStep','fields':{'simulation_time':float(state[0]),'state':state.tobytes(),'torque':command.tobytes(),
        'warnings':{'dataclass':'WarningLedger','fields':{'counts':[0]*8,'lastinfo':[0]*8}}}}
    report['session']['stepper'].update(attempted=3,returned=3,captured=3,capture_attempts=3,expected_time=float(state[0]))
    report['session']['foundation'].update(attempted=3,returned=3,unexecuted=18187)
    report['api_counters'].update(step_attempted=3,step_returned=3)
    report['first_error']={'type':'StageTimeout'};metadata['summary']=copy.deepcopy(report['session'])
    return a,metadata,report,capsules,initial,contract,{'adapter':{'last_capture':value}}

class SidecarIO(unittest.TestCase):
    def test_bound_complete_sidecars(self):
        f=fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);pins=write_case(p,f);r,_=read_sidecars(p,f['report'],f['stages'],f['arrays'],pins)
            self.assertTrue(r['sidecar_evidence_integrity_passed'] and r['instrumentation_complete'])
    def test_unpinned_or_altered_bytes_rejected(self):
        for mutation in ('pin','bytes','size'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
                f=fixture();p=Path(d);pins=write_case(p,f)
                if mutation=='pin':pins.pop((p/'timing_gc.bin').resolve())
                elif mutation=='bytes':(p/'timing_spans.bin').write_bytes(b'changed')
                else:f['report']['timing_instrumentation']['files']['timing_spans.bin']['bytes']+=1
                with self.assertRaises(AssertionError):read_sidecars(p,f['report'],f['stages'],f['arrays'],pins)
    def test_failed_sidecar_writer_retains_partial_without_qualification(self):
        f=fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);pins=write_case(p,f);decl=f['report']['timing_instrumentation']
            decl['files'].pop('timing_spans.bin');decl.update(complete=False,errors=[{'phase':'spans','type':'OSError','detail':'injected'}])
            r,_=read_sidecars(p,f['report'],f['stages'],f['arrays'],pins)
            self.assertTrue(r['sidecar_evidence_integrity_passed']);self.assertFalse(r['sidecar_available']);self.assertFalse(r['instrumentation_complete'])
    def test_missing_sidecar_cannot_claim_complete(self):
        f=fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);pins=write_case(p,f);f['report']['timing_instrumentation']['files'].pop('timing_gc.bin')
            with self.assertRaises(AssertionError):read_sidecars(p,f['report'],f['stages'],f['arrays'],pins)
    def test_actual_reserved_capture_without_verification_credit(self):
        args=reserved();r=reserved_capture(*args)
        self.assertTrue(r['actual_full291_checked'] and r['strict_captured_state_passed'])
        self.assertFalse(r['actual_native_verification_returned']);self.assertEqual(r['new_verification_credit'],0)
        self.assertFalse(r['full_scope_credit'])
    def test_counter_gap_requires_prior_independent_proof(self):
        a,meta,report,capsules,*_=reserved()
        with self.assertRaises(AssertionError):counts(a,meta,report,capsules)
        r=counts(a,meta,report,capsules,{'partial_counter_proof':True})
        self.assertEqual(r['actual_native_returned'],3);self.assertEqual(r['foundation']['captured'],2)
        self.assertFalse(r['complete_native_coverage'])
    def test_reserved_pd_or_time_mutation_rejected(self):
        for key,index in [('ctrl',89),('clock',0),('qpos',291)]:
            with self.subTest(key=key):
                args=list(reserved());f=args[-1]['adapter']['last_capture']['fields']
                state=np.frombuffer(f['state'],np.float64).copy();state[index]+=.1;f['state']=state.tobytes()
                with self.assertRaises(AssertionError):reserved_capture(*args)
    def test_reserved_warning_overflow_not_silently_cast(self):
        args=list(reserved());args[-1]['adapter']['last_capture']['fields']['warnings']['fields']['counts'][0]=1<<32
        with self.assertRaises((AssertionError,OverflowError)):reserved_capture(*args)
    def test_literal_source_lineage_and_all32_actual_pins(self):
        new=Path(__file__).resolve().parents[2];current=new/'independent_plant_timing_integration_v1/source_draft_v1'
        prior=new/'independent_plant_pending_result_v1/source_draft_v1'
        roles={'clock_runner':current/'run_clock.py','clock_core':current/'clock_core.py','clock_watchdog':current/'stage_watchdog.py',
            'clock_worker':current/'dummy_worker.py','clock_protocol':current/'recorded_protocol.py','clock_pending_result':current/'pending_result.py',
            'timing_source_preparation':current.parent/'source_preparation.json','timing_source_review':new/'independent_timing_integration_root_review_v1/review.json'}
        for key in old_stage.SOURCE_SHA:
            if key=='original_core':p=new/'independent_plant_clock_timeout_correction_v1/source_draft_v1/clock_core.py'
            elif key=='original_runner':p=new/'independent_plant_process_clock_v1/source_runner_v1/run_clock.py'
            else:p=prior/roles[key].name
            roles['prior_'+key]=p
        pins={p.resolve():source.digest(p) for p in current.glob('*.py')}
        self.assertTrue(source.source_contract(roles,pins)['all32_source_pins_exact'])
        pins.pop((current/'timing_hooks.py').resolve())
        with self.assertRaises(AssertionError):source.source_contract(roles,pins)
    def test_explicit_timing_contract_required(self):
        with self.assertRaises(AssertionError):source.request_contract({})
        source.request_contract({'timing_probe_contract':source.CONTRACT})


if __name__=='__main__':unittest.main()
