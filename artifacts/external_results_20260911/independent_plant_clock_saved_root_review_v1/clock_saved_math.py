"""Independent saved-array arithmetic; no runtime, model, worker or native imports."""
import numpy as np

DT_NS = 2_000_000
TOTAL_STEPS = 18190
TOTAL_CONTROLS = 1819


def exact(actual, expected, name):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
        raise AssertionError(name + ': shape, dtype or bytes')


def array(value, dtype, shape, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape:
        raise AssertionError(name + ': schema')
    if not np.isfinite(value).all():
        raise AssertionError(name + ': finite values')
    return value


def repeated_time(initial, n):
    current = float(initial)
    output = np.empty(n, np.float64)
    for i in range(n):
        current += .002
        output[i] = current
    return output


def first_difference(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise AssertionError('Compared prefix schema differs')
    if len(a) == 0:
        return None
    # Compare bytes, preserving signed-zero and every original saved bit.
    av = np.ascontiguousarray(a).view(np.uint8).reshape(len(a), -1)
    bv = np.ascontiguousarray(b).view(np.uint8).reshape(len(b), -1)
    rows = np.flatnonzero(np.any(av != bv, axis=1))
    return None if not len(rows) else int(rows[0])


def physical_arrays(a, initial, contract):
    """Check all captured actual commands/states, including a failed/held prefix."""
    n = len(a['packed_capture'])
    if not 0 <= n <= TOTAL_STEPS:
        raise AssertionError('Original full requested scope')
    initial = array(initial, np.float64, (291,), 'Initial full291')
    packed = array(a['packed_capture'], np.float64, (n, 373), 'Packed373')
    schema = {'step_integration':291, 'step_qpos':30, 'step_qvel':29, 'step_actuator_force':23,
              'step_command':23, 'step_target':23}
    for key, width in schema.items():
        array(a[key], np.float64, (n, width), key)
    array(a['step_raw_action'], np.float32, (n,23), 'Raw action')
    for key in ('step_warning_counts', 'step_warning_lastinfo'):
        array(a[key], np.int32, (n,8), key)
    for key, expected in [('step_integration', packed[:,:291]), ('step_qpos', packed[:,291:321]),
                          ('step_qvel', packed[:,321:350]), ('step_actuator_force', packed[:,350:])]:
        exact(a[key], expected, key+' packed slices')
    integration, q, v = a['step_integration'], a['step_qpos'], a['step_qvel']
    exact(integration[:,1:31], q, 'Integration position overlap')
    exact(integration[:,31:60], v, 'Integration velocity overlap')
    exact(integration[:,89:112], a['step_command'], 'Actual integration ctrl')
    exact(integration[:,0], repeated_time(initial[0],n), 'Unreset repeated2ms clock')
    exact(a['step_expected_simulation_time'], integration[:,0], 'Saved expected clock')
    if np.any(integration[:,112:]):
        raise AssertionError('Unexpected applied/root external force')
    previous_q = np.concatenate([initial[None,8:31], q[:-1,7:]], axis=0)[:n]
    previous_v = np.concatenate([initial[None,37:60], v[:-1,6:]], axis=0)[:n]
    pd = np.clip(contract['kp']*(a['step_target']-previous_q)-contract['kd']*previous_v,
                 -contract['native_effort'], contract['native_effort'])
    exact(a['step_command'], pd, 'Independent actual manual PD')
    limits = contract['joint_limits']
    if np.any(a['step_target'] < limits[:,0]) or np.any(a['step_target'] > limits[:,1]):
        raise AssertionError('Applied target exceeds original native bounds')
    excess = np.maximum(np.maximum(limits[:,0]-q[:,7:], q[:,7:]-limits[:,1]),0.).max(axis=1)
    speed = (np.abs(v[:,6:])/contract['native_velocity']).max(axis=1)
    effort = (np.abs(a['step_actuator_force'])/contract['native_effort']).max(axis=1)
    tilt = np.arccos(np.clip(1-2*(q[:,4]**2+q[:,5]**2),-1,1))
    bad = dict(joint_bound=excess>1e-6, joint_speed=speed>1., effort=effort>1+1e-9,
        quaternion=np.abs(np.linalg.norm(q[:,3:7],axis=1)-1)>1e-10,
        fall=(q[:,2]<.25)|(tilt>1.2), warning=np.any(a['step_warning_counts']!=0,axis=1))
    failed = np.zeros(n, bool)
    for value in bad.values(): failed |= value
    array(a['step_verified'], np.bool_, (n,), 'Saved verification mask')
    exact(a['step_verified'], ~failed, 'Independent original strict predicates')
    return dict(captured_steps=n, strict_failures={k:np.flatnonzero(v).tolist() for k,v in bad.items()},
                all_actual_saved_states_strict=not bool(failed.any()), full_scope=n==TOTAL_STEPS)


def timing_arrays(a, outer_records, epoch_ns):
    if type(epoch_ns) is not int or epoch_ns < 0:
        raise AssertionError('Literal monotonic integer epoch')
    n = len(a['step_index'])
    exact(a['step_index'], np.arange(n,dtype=np.int64), 'Every actual native index')
    starts = epoch_ns + np.arange(n,dtype=np.int64)*DT_NS
    ends = starts+DT_NS
    exact(a['step_nominal_start'], starts, 'Fixed nominal starts')
    exact(a['step_nominal_end'], ends, 'Fixed nominal ends')
    for key in ('step_actual_start','step_actual_end','step_wake_lateness_ns','step_finish_lateness_ns','step_wake_debt'):
        array(a[key],np.int64,(n,),key)
    actual_start, actual_end = a['step_actual_start'], a['step_actual_end']
    if np.any(actual_start < starts) or np.any(actual_end < actual_start):
        raise AssertionError('Impossible actual timing order')
    if n>1 and np.any(actual_start[1:] < actual_end[:-1]):
        raise AssertionError('Overlapping serial native cycles')
    exact(a['step_wake_lateness_ns'], actual_start-starts, 'Wake lateness')
    exact(a['step_finish_lateness_ns'], np.maximum(actual_end-ends,0), 'Finish lateness')
    expected_debt = np.maximum(0, np.minimum(TOTAL_STEPS,(actual_start-epoch_ns)//DT_NS)-np.arange(n))
    exact(a['step_wake_debt'], expected_debt.astype(np.int64), 'Fixed epoch wake debt')
    outer_indices, outer_misses = [], []
    for record in outer_records:
        i = record['index']
        if type(i) is not int or not 0<=i<n or i in outer_indices:
            raise AssertionError('Outer committed index schema')
        outer_indices.append(i)
        if record['returned'] != i+1 or record['fixed_nominal_end_ns'] != int(ends[i]):
            raise AssertionError('Outer original deadline/counter')
        if type(record['cycle_return_ns']) is not int or record['cycle_return_ns'] < int(actual_end[i]):
            raise AssertionError('Outer time precedes verified native return')
        if record['cycle_return_ns'] > int(ends[i]): outer_misses.append(i)
    if outer_indices != list(range(len(outer_indices))):
        raise AssertionError('Outer ledger skips/reorders committed steps')
    return dict(captured_step_deadline_misses=np.flatnonzero(actual_end>ends).tolist(),
        outer_deadline_misses=outer_misses, outer_rows=len(outer_indices),
        complete_outer_coverage=len(outer_indices)==n,
        fixed_epoch_timing_pass=n==TOTAL_STEPS and len(outer_indices)==n and not np.any(actual_end>ends) and not outer_misses)


def compare_expert_prefix(a, full, hold):
    """Report actual differences honestly; held commands can change later physics."""
    n = len(a['step_index'])
    exact(full['final_integration'], hold['initial_integration'], 'Original continuous hold')
    fields = {}
    for actual, source, skip_initial in [
        ('step_qpos','physics_qpos',True), ('step_qvel','physics_qvel',True),
        ('step_command','physics_torque',False), ('step_warning_lastinfo','physics_warning_lastinfo',True)]:
        expected = np.concatenate([z[source][1:] if skip_initial else z[source] for z in (full,hold)])
        fields[actual] = first_difference(a[actual], expected[:n])
    for actual, choices, skip in [('step_actuator_force',('physics_actuator_force','physics_actuator_torque'),False),
                                  ('step_warning_counts',('physics_warning_number','physics_warning_counts'),True)]:
        pieces = [z[next(k for k in choices if k in z)] for z in (full,hold)]
        expected = np.concatenate([v[1:] if skip else v for v in pieces])
        fields[actual] = first_difference(a[actual],expected[:n])
    for key in ('target','raw_action'):
        source = 'target' if key=='target' else 'action'
        expected = np.concatenate([np.repeat(z[source],z['physics_substeps'],axis=0) for z in (full,hold)])
        fields['step_'+key] = first_difference(a['step_'+key],expected[:n])
    return dict(first_difference_by_field=fields, original_expert_prefix_exact=all(v is None for v in fields.values()),
                full_expert_scope=n==TOTAL_STEPS,
                limitation='A differing actual command fails recorded-command qualification; subsequent state differences are retained, not replaced with expert states.')
