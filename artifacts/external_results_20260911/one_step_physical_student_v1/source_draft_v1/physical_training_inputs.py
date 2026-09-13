"""Read-only binding and fixed physical-row selection; no model/physics imports."""
from pathlib import Path
import hashlib
import json
import numpy as np
from physical_training_objective import physical_cells, REQUESTED_COUNTS, expected_budgets

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def exact(a, b, name):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), name

def contains_hash(value, digest):
    if isinstance(value, dict):
        return any(contains_hash(v, digest) for v in value.values())
    if isinstance(value, list):
        return any(contains_hash(v, digest) for v in value)
    return value == digest

def expected_identities():
    dataset=np.repeat(np.arange(3,dtype=np.int64),1018)
    control=np.tile(np.arange(250,1268,dtype=np.int64),3)
    index=dataset*1019+control-250
    return dataset,control,index

def resolve_array_path(manifest_path, name):
    root=Path(manifest_path).parent.resolve()
    path=(root/name).resolve()
    assert path.parent==root, name
    return path

def validate_status_vectors(valid, nominal_verified, nominal_status, policy_status):
    assert valid.dtype==np.bool_ and nominal_verified.dtype==np.bool_
    assert valid.ndim==1 and valid.shape==nominal_verified.shape==nominal_status.shape==policy_status.shape
    assert nominal_verified.all() and (nominal_status==1).all()
    assert np.array_equal(valid,policy_status==1)
    assert (policy_status[~valid]==2).all()

def validate_bound_review(spec, input_pins):
    path = spec['path']
    assert input_pins[path] == spec['sha256'] == sha(path), path
    receipt = read(path)
    assert receipt[spec['pass_field']] is True, path
    assert spec['subjects'], path
    for subject, digest in spec['subjects'].items():
        assert input_pins[subject] == digest == sha(subject), subject
        assert contains_hash(receipt, digest), (path, subject)

def load_physical(request, centers):
    """Use final collector schema only. Requested identities never disappear."""
    paths = request['physical_paths']
    manifest_path = Path(paths['manifest'])
    manifest, report = read(manifest_path), read(paths['report'])
    assert report['completed'] is True and report['passed'] is True
    assert report['rows'] == report['nominal_verified'] == report['policy_branches'] == 3054
    assert manifest['complete'] is True and manifest['rows'] == 3054
    assert report['manifest_sha256'] == sha(manifest_path)
    assert report['request_sha256'] == manifest['request_sha256'] == sha(paths['collection_request'])
    assert report['query250_actual251_calibration_passed'] is True
    assert report['all_frozen_inputs_unchanged'] is True
    arrays = {}
    for key, spec in manifest['arrays'].items():
        p = resolve_array_path(manifest_path,spec['path'])
        assert sha(p) == spec['sha256'] == request['input_sha256'][p.as_posix()], key
        a = np.load(p, mmap_mode='r', allow_pickle=False)
        assert list(a.shape) == spec['shape'] and str(a.dtype) == spec['dtype'], key
        assert len(a) == 3054, key
        arrays[key] = a
    d,c,index=expected_identities()
    for key, value in dict(dataset=d, start_control=c, successor_control=c+1,
            source_frame=c+11, successor_frame=c+12, center_index=index,
            successor_center_index=index+1).items():
        exact(arrays[key], value, key)
    exact(arrays['nominal_features'], centers['features'][index], 'original nominal inputs')
    exact(arrays['nominal_base_target'], centers['base_target'][index], 'original nominal base')
    exact(arrays['nominal_target'], centers['expert_target'][index], 'original nominal target')
    assert arrays['nominal_verified'].dtype == np.bool_ and arrays['nominal_verified'].all()
    assert (arrays['nominal_status'] == 1).all()
    for key in ('valid_steps', 'attempted_steps', 'returned_steps'):
        assert (arrays['nominal_' + key] == 10).all()
    valid = arrays['label_valid']
    assert valid.dtype == np.bool_ and valid.shape == (3054,)
    validate_status_vectors(valid,arrays['nominal_verified'],arrays['nominal_status'],arrays['policy_status'])
    for key in ('valid_steps', 'attempted_steps', 'returned_steps'):
        assert (arrays['policy_' + key][valid] == 10).all()
        assert ((arrays['policy_' + key][~valid] >= 1) & (arrays['policy_' + key][~valid] <= 10)).all()
    assert report['labels_valid'] == manifest['labels_valid'] == int(valid.sum()) == request['valid_rows']
    assert report['strict_failed'] == manifest['strict_failed'] == int((~valid).sum())
    assert report['labels_valid'] > 0, 'No new physical supervision; no fit selected for an empty dataset'
    expected_calls = dict(backward=int(valid.sum()), actor=int(valid.sum()), head=3054+int(valid.sum()))
    assert report['graph_calls'] == {k:dict(attempted=v, returned=v) for k,v in expected_calls.items()}
    assert manifest['graph_calls'] == report['graph_calls']
    assert report['native_attempted'] == report['native_returned'] <= 61080
    assert report['native_returned'] == 30540 + int(arrays['policy_returned_steps'].sum())
    selected = np.flatnonzero(valid)
    out = dict(rows=selected, valid_mask=np.array(valid), dataset=d[selected],
        start_control=c[selected], successor_control=c[selected]+1, successor=index[selected]+1,
        all_dataset=d, all_successor_control=c+1)
    for name, key, shape, dtype in (
        ('features','endpoint_features',(1069,),np.float32),
        ('base','endpoint_base_target',(23,),np.float64),
        ('target','label_fixed_map_target',(23,),np.float64),
        ('residual','label_residual_rad',(23,),np.float64)):
        value = arrays[key]
        assert value.shape == (3054,*shape) and value.dtype == dtype, key
        assert np.isfinite(value[valid]).all() and np.isnan(value[~valid]).all(), key
        out[name] = np.asarray(value[selected]).copy()
    exact(out['residual'], out['target']-out['base'], 'original branch target-minus-base labels')
    lo, hi = centers['joint_limits'].T
    assert ((out['target'] >= lo) & (out['target'] <= hi)).all()
    out['cells'] = physical_cells(out['dataset'], out['successor_control'])
    requested = physical_cells(d, c+1)
    assert tuple(len(ids) for ids in requested) == REQUESTED_COUNTS
    out['coverage'] = [dict(dataset=cell//3, phase=('acquisition','source','return')[cell%3],
        requested=n, valid=len(ids), strict_failed=n-len(ids), effective_mass=len(ids)/(9*n),
        empty=len(ids)==0) for cell,(ids,n) in enumerate(zip(out['cells'], REQUESTED_COUNTS))]
    for key in ('policy_native_target_clipped','teacher_feedback_clipped','teacher_native_clipped',
                'successor_replan_boundary','successor_zero_gain'):
        out[key] = np.asarray(arrays[key][selected]).copy()
    out['collector_head_delta']=np.asarray(arrays['endpoint_head_delta'][selected]).copy()
    assert out['collector_head_delta'].shape==(len(selected),23) and out['collector_head_delta'].dtype==np.float32
    assert np.isfinite(out['collector_head_delta']).all()
    out['budgets'] = expected_budgets(len(selected))
    assert request['budgets'] == out['budgets']
    return out
