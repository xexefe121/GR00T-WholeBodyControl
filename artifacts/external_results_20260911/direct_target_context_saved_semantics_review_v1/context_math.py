"""Independent incoming-context composition; saved arrays only, no producer imports."""
from saved_common import np, phase, flat

WIDTHS = {'actions':23, 'base_ang_vel':3, 'dof_pos':23, 'dof_vel':23, 'projected_gravity':3}


def context(previous, history):
    assert previous.dtype == np.float32 and previous.shape == (23,)
    assert np.isfinite(previous).all() and set(history) == set(WIDTHS)
    for name, width in WIDTHS.items():
        value = history[name]
        assert value.dtype == np.float32 and value.shape == (4, width)
        assert np.isfinite(value).all()
    # flat() keeps actions/base_ang_vel/dof_pos/dof_vel/projected_gravity order.
    return np.concatenate((previous, flat(history)))


def features(current, previous, history, control):
    assert current.dtype == np.float32 and current.shape == (1000,)
    assert np.isfinite(current).all()
    incoming = context(previous, history)
    if phase(control) == 2:
        return np.zeros(1323, np.float32)
    return np.concatenate((current, incoming))


def nominal_features(original, previous, history):
    assert original.ndim == previous.ndim == history.ndim == 2
    n = len(original)
    assert original.shape == (n,1069) and previous.shape == (n,23) and history.shape == (n,300)
    assert all(value.dtype == np.float32 and np.isfinite(value).all()
               for value in (original,previous,history))
    kept = np.r_[np.arange(52),np.arange(75,1023)]
    return np.concatenate((original[:,kept],previous,history),axis=1)


def map_selection(centers, selected):
    """Never synthesize or choose between maps. Missing maps are explicitly uncovered."""
    chosen, missing = [], []
    for control in selected:
        matches = np.flatnonzero((centers['dataset']==2)&(centers['control']==control))
        assert len(matches) <= 1, 'Ambiguous same-clock query250 map '+str(control)
        if len(matches) == 0:
            missing.append(int(control))
            continue
        index = int(matches[0])
        assert int(centers['plan_control'][index])+int(centers['plan_local'][index]) == int(control)
        assert centers['gain'][index].shape == (23,58)
        assert all(np.isfinite(centers[name][index]).all()
                   for name in ('gain','planned_state','planned_target','qpos','qvel','expert_target'))
        chosen.append(int(control))
    return np.asarray(chosen,np.int64), missing
