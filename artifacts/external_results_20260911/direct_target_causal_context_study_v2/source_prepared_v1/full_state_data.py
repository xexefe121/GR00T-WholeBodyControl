"""Extend the byte-preserved9904-anchor loader with qualified58-axis data."""
from pathlib import Path
import numpy as np
from direct_data import load_data, read, sha
from full_state_contract import center_cells

def load_full_state_data(original_paths, full_paths, pins):
    data = load_data(original_paths, pins)
    for path in full_paths.values():
        canonical = Path(path).as_posix()
        if pins.get(canonical) != sha(path):
            raise ValueError('Full-state consumed path is not frozen: '+canonical)
    report = read(full_paths['report'])
    if report['complete'] is not True or report['all_slots_valid'] is not True:
        raise ValueError('Full-state generation incomplete.')
    if report['signed_rows'] != 354612 or report['overlap_rows'] != 140622 or report['new_rows'] != 213990:
        raise ValueError('Unexpected fixed full-state counts.')
    if report['original_center_and_overlap_exact'] is not True or report['all_inputs_unchanged'] is not True:
        raise ValueError('Full-state generation parity failed.')
    if report['request_sha256'] != sha(full_paths['request']):
        raise ValueError('Full-state generation request mismatch.')
    parent = Path(full_paths['report']).parent
    manifest_path=parent/'manifest.json'
    if pins.get(manifest_path.as_posix())!=report['manifest_sha256'] or sha(manifest_path)!=report['manifest_sha256']:
        raise ValueError('Full-state manifest is not frozen.')
    manifest=read(manifest_path)
    if manifest['complete'] is not True or manifest['request_sha256']!=report['request_sha256']:
        raise ValueError('Full-state manifest subject differs.')
    # Every source array from generation is bound, even if not a training input.
    for filename, digest in report['output_sha256'].items():
        path = (parent / filename).resolve()
        if path.parent != parent.resolve() or pins.get(path.as_posix()) != digest or sha(path) != digest:
            raise ValueError('Full-state output not frozen exactly: '+filename)
    with np.load(parent/'centers.npz', allow_pickle=False) as z:
        centers = {key:z[key].copy() for key in z.files}
    cells = center_cells(centers['dataset'], centers['control'], centers['axis_group'])
    expected_phase = np.tile(np.repeat(np.arange(3,dtype=np.int8),[100,819,100]),3)
    expected_cell = np.repeat(np.arange(9,dtype=np.int8),[100,819,100]*3)
    checks = dict(features=data['features'][data['center_map']], target=data['target'][data['center_map']],
        dataset=data['center_dataset'],control=data['center_control'],source_frame=data['center_control']+11,
        phase=expected_phase,cell=expected_cell,joint_span=data['span'],joint_limits=data['limits'])
    for key, expected in checks.items():
        actual = centers[key]
        if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
            raise ValueError('Full-state center bytes differ: '+key)
    radii = centers['axis_radius']
    if radii.shape != (58,) or radii.dtype != np.float64 or not np.isfinite(radii).all() or np.any(radii<=0):
        raise ValueError('Full-state radii schema.')
    arrays = {}
    for key, width, dtype in (
        ('features',1000,np.float32),('target',23,np.float64),('target_change',23,np.float64),
        ('feedback_clipped',23,np.bool_),('native_clipped',23,np.bool_),('signed_radius',None,np.float64),('status',None,np.uint8)):
        array = np.load(parent/(key+'.npy'),mmap_mode='r',allow_pickle=False)
        expected_shape = (3057,58,2) + (() if width is None else (width,))
        if array.shape != expected_shape or array.dtype != dtype:
            raise ValueError('Full-state array schema: '+key)
        spec=manifest['arrays'][key]
        if spec['path']!=key+'.npy' or spec['shape']!=list(array.shape) or spec['dtype']!=str(array.dtype) or spec['sha256']!=report['output_sha256'][key+'.npy']:
            raise ValueError('Full-state manifest schema/identity differs: '+key)
        arrays[key] = array
    if not np.all(arrays['status']==1):
        raise ValueError('Every full-state slot must be verified; no masking or redistribution.')
    if not np.array_equal(arrays['signed_radius'], np.broadcast_to(radii[None,:,None]*np.array([-1.,1.])[None,None,:],(3057,58,2))):
        raise ValueError('Signed radius changes or adaptive shrinking are not selected.')
    features = arrays['features'].reshape(354612,1000)
    target = arrays['target'].reshape(354612,23)
    changes = arrays['target_change'].reshape(354612,23)
    for start in range(0,354612,8192):
        stop = min(start+8192,354612)
        if not all(np.isfinite(array[start:stop]).all() for array in (features,target,changes)):
            raise ValueError('Nonfinite full-state values.')
        if not ((target[start:stop]>=data['limits'][:,0])&(target[start:stop]<=data['limits'][:,1])).all():
            raise ValueError('Full-state teacher target violates native bounds.')
        index = np.arange(start,stop,dtype=np.int64)//116
        expected = target[start:stop]-centers['target'][index]
        if changes[start:stop].tobytes() != expected.tobytes():
            raise ValueError('Target difference bytes differ from actual teacher targets.')
    data.update(full_state_features=features,full_state_target=target,full_state_target_change=changes,
                full_state_centers=centers,full_state_cells=cells,
                full_state_group_axes=[np.flatnonzero(centers['axis_group']==g) for g in range(6)],
                full_state_flags={key:arrays[key] for key in ('feedback_clipped','native_clipped')},
                full_state_generation_report=report)
    return data
