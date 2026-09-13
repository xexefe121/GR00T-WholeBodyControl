"""Independent saved-array math. No training/runtime imports or model calls."""
import numpy as np

DATASET_COUNTS = ((100,819,100),(100,819,100),(100,819,100),(100,5780,100),(100,667,100))


def phase_indices(counts=DATASET_COUNTS):
    result = []
    cursor = 0
    for phases in counts:
        for count in phases:
            result.append(np.arange(cursor, cursor + count, dtype=np.int64))
            cursor += count
    return result


def balanced_moments(features, groups):
    if features.dtype != np.float32 or features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError('finite float32 matrix required')
    weight = np.zeros(len(features), dtype=np.float64)
    for group in groups:
        if len(group) == 0 or len(np.unique(group)) != len(group) or np.any(weight[group]):
            raise ValueError('groups must be nonempty and disjoint')
        weight[group] = 1.0 / (len(groups) * len(group))
    if np.any(weight == 0):
        raise ValueError('every row must have a group')
    wide = features.astype(np.float64)
    # Direct weighted row sums, independently of producer's cell means.
    mean = np.sum(wide * weight[:,None], axis=0)
    variance = np.sum((wide - mean)**2 * weight[:,None], axis=0)
    return mean, variance, mean.astype(np.float32), np.maximum(np.sqrt(variance), .05).astype(np.float32)


def targets(output, default, span, limits):
    if output.dtype != np.float32 or default.dtype != np.float64 or span.dtype != np.float32 or limits.dtype != np.float64:
        raise ValueError('explicit output cast contract required')
    if output.ndim != 2 or output.shape[1] != 23 or not np.isfinite(output).all():
        raise ValueError('finite float32 Nx23 output required')
    raw = default[None,:] + output.astype(np.float64) * span.astype(np.float64)[None,:]
    applied = np.minimum(np.maximum(raw, limits[:,0]), limits[:,1])
    error = raw - applied
    return raw, applied, {'clipped_rows':int(np.count_nonzero(np.any(error != 0, axis=1))),
                         'clipped_components':int(np.count_nonzero(error))}


def position_errors(output, teacher, default, span, limits):
    raw, applied, clipping = targets(output, default, span, limits)
    return dict(preclip_RMSE_rad=float(np.sqrt(np.average(np.square(raw-teacher)))),
                applied_RMSE_rad=float(np.sqrt(np.average(np.square(applied-teacher)))),
                preclip_per_joint_RMSE_rad=np.sqrt(np.average(np.square(raw-teacher),axis=0)).tolist(),
                applied_per_joint_RMSE_rad=np.sqrt(np.average(np.square(applied-teacher),axis=0)).tolist(), **clipping)


def response_error(endpoint, nominal, endpoint_teacher, nominal_teacher, span):
    # Cast BEFORE subtraction; defaults cancel, BFM bases never enter.
    expected_change = np.subtract(endpoint_teacher, nominal_teacher) / span.astype(np.float64)
    actual_change = np.subtract(endpoint.astype(np.float64), nominal.astype(np.float64))
    return actual_change - expected_change


def summarize(nominal, velocity, physical, data):
    default, span, limits = data['default'], data['span'], data['limits']
    groups = phase_indices()
    nominal_labels = ((data['nominal_teacher']-default)/span.astype(np.float64)).astype(np.float32)
    # Producer computes f32 squares/reduction; wider reduction is compared with
    # an explicit roundoff tolerance by caller, never claimed byte-identical.
    nominal_square = np.square(nominal-nominal_labels).astype(np.float64)
    n_cells, v_cells, p_cells = [], [], []
    for index, group in enumerate(groups):
        n_cells.append(dict(dataset=index//3, phase=index%3, rows=len(group),
            normalized_MSE=float(np.mean(nominal_square[group])),
            first24=position_errors(nominal[group[:24]],data['nominal_teacher'][group[:24]],default,span,limits),
            **position_errors(nominal[group],data['nominal_teacher'][group],default,span,limits)))
    v = velocity.reshape(3057,23,2,23)
    v_error = response_error(v,nominal[:3057,None,None,:],data['velocity_teacher'].reshape(v.shape),
                             data['nominal_teacher'][:3057,None,None,:],span)
    for index, group in enumerate(groups[:9]):
        rows = (group[:,None]*46 + np.arange(46)[None,:]).reshape(-1)
        v_cells.append(dict(dataset=index//3,phase=index%3,center_rows=len(group),probe_rows=len(rows),
            response_MSE=float(np.mean(v_error[group]**2)),first24_response_MSE=float(np.mean(v_error[group[:24]]**2)),
            **position_errors(velocity[rows],data['velocity_teacher'][rows],default,span,limits)))
    successor = data['physical_successor']
    p_error = response_error(physical,nominal[successor],data['physical_teacher'],data['nominal_teacher'][successor],span)
    p_groups = phase_indices(((99,819,100),)*3)
    for index, group in enumerate(p_groups):
        p_cells.append(dict(dataset=index//3,phase=index%3,requested=len(group),valid=len(group),
            response_MSE=float(np.sum(p_error[group]**2)/(len(group)*23)),
            first24_response_MSE=float(np.mean(p_error[group[:24]]**2)),
            **position_errors(physical[group],data['physical_teacher'][group],default,span,limits)))
    extra = {}
    for name, flag in data.get('physical_flags',{}).items():
        mask = np.any(flag,axis=1) if flag.ndim == 2 else flag
        extra[name] = dict(rows=int(mask.sum()),response_MSE=float(np.mean(p_error[mask]**2)) if mask.any() else None)
    return dict(nominal_objective=float(np.mean([c['normalized_MSE'] for c in n_cells])),
        full_velocity_objective=float(np.mean([c['response_MSE'] for c in v_cells])),
        physical_objective=float(np.mean([c['response_MSE'] for c in p_cells])),
        nominal_cells=n_cells,velocity_cells=v_cells,physical_cells=p_cells,physical_groups=extra,
        no_checkpoint_selection=True,no_connected_stability_claim=True)
