"""Independent numerical gate and all saved drift checks; no inference."""
import json
from pathlib import Path
import numpy as np
from audit_graph import exact

CORPORA = {'nominal':9904, 'velocity':140622, 'physical':3054}
NEW = ('CPU64','GPU64','ORT64')
OLD = ('GPU','CPU','ORT')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def measure_outputs(directory, old_fit, norm, bind):
    directory, old_fit = Path(directory), Path(old_fit)
    default, span, limits = [norm[k] for k in ('default_q','joint_span','joint_limits')]
    assert default.dtype == limits.dtype == np.float64 and span.dtype == np.float32
    assert default.shape == span.shape == (23,) and limits.shape == (23,2)
    def load(path, count):
        value = np.load(bind(path), allow_pickle=False)
        assert value.dtype == np.float32 and value.shape == (count,23) and np.isfinite(value).all()
        raw = default[None,:] + value.astype(np.float64) * span.astype(np.float64)[None,:]
        applied = np.minimum(np.maximum(raw, limits[:,0]), limits[:,1])
        return value, raw, raw != applied
    outputs = {b:{} for b in NEW}
    new = {b:{} for b in NEW}
    old = {b:{} for b in OLD}
    for backend in NEW:
        for corpus, count in CORPORA.items():
            values, raw, mask = load(directory / (backend+'_'+corpus+'.npy'), count)
            outputs[backend][corpus] = values
            new[backend][corpus] = (raw,mask)
    for backend in OLD:
        for corpus, count in CORPORA.items():
            _, raw, mask = load(old_fit / ('final_'+backend+'_'+corpus+'.npy'), count)
            old[backend][corpus] = (raw,mask)
    comparisons = {}
    for corpus in CORPORA:
        for left,right,key in [('GPU64','CPU64','GPU_CPU'),('GPU64','ORT64','GPU_ORT'),('CPU64','ORT64','CPU_ORT')]:
            comparisons[corpus+'_'+key] = float(np.max(np.abs(new[left][corpus][0]-new[right][corpus][0])))
    numerical = read(bind(directory/'parity.json'))
    assert numerical['comparisons'] == comparisons
    maximum = max(comparisons.values())
    assert np.isfinite(maximum) and maximum == numerical['maximum_preclamp_rad']
    assert numerical['tolerance_rad'] == 1e-5 and numerical['nonfinite_comparisons'] == []
    assert numerical['passed'] is (maximum <= 1e-5)
    drift_report = read(bind(directory/'drift.json'))
    verified_drifts = {}
    for backend in NEW:
        for previous in OLD:
            for corpus in CORPORA:
                filename = 'drift_'+backend+'_vs_'+previous+'32_'+corpus+'.npy'
                raw, mask = new[backend][corpus]
                old_raw, old_mask = old[previous][corpus]
                difference = raw - old_raw
                actual = np.load(bind(directory/filename), allow_pickle=False)
                exact(actual, difference, filename)
                changed = mask != old_mask
                expected = dict(max_abs_preclamp_rad=float(np.max(np.abs(difference))),
                    RMS_preclamp_rad=float(np.sqrt(np.mean(difference**2))), shape=list(difference.shape),
                    old_clipped_rows=int(np.count_nonzero(np.any(old_mask,axis=1))),
                    old_clipped_components=int(np.count_nonzero(old_mask)),
                    new_clipped_rows=int(np.count_nonzero(np.any(mask,axis=1))),
                    new_clipped_components=int(np.count_nonzero(mask)),
                    clipping_changed_rows=int(np.count_nonzero(np.any(changed,axis=1))),
                    clipping_changed_components=int(np.count_nonzero(changed)))
                assert drift_report[filename] == expected, filename+' statistics'
                verified_drifts[filename] = expected
    assert set(drift_report) == set(verified_drifts) and len(verified_drifts) == 27
    ledger_path = bind(directory/'graph_calls.jsonl')
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    expected_ledger = []
    for backend in NEW:
        for corpus,count in CORPORA.items():
            for start in range(0,count,256):
                expected_ledger.append(dict(backend=backend,corpus=corpus,start=start,stop=min(start+256,count),
                    returned=True,synchronized=True,verified=True))
    assert len(ledger) == 1803 and ledger == expected_ledger
    counters = {backend:dict(calls_attempted=601,calls_returned=601,calls_synchronized=601,calls_verified=601,
                            rows_attempted=153580,rows_returned=153580,rows_verified=153580) for backend in NEW}
    return dict(parity_passed=maximum<=1e-5,maximum_preclamp_rad=maximum,comparisons=comparisons,
                drift_arrays_verified=27,drifts=verified_drifts,counters=counters,
                all_call_partitions_exact=True,model_calls=0), outputs
