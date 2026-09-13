"""Independent binary and typed-JSON decoding; NumPy/stdlib only."""
import base64
import hashlib
import json
import math
import numpy as np


def exact(actual, expected, name):
    a,b=np.asarray(actual),np.asarray(expected)
    if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():
        raise AssertionError(name+': dtype, shape or bytes differ')


def unpack373(packed):
    """Owned little-endian floats; no adapter capture/decode imports."""
    if not isinstance(packed,np.ndarray) or packed.dtype!=np.uint8 or packed.ndim!=2 or packed.shape[1]!=2984:
        raise ValueError('uint8[N,2984] capture required')
    values=np.frombuffer(packed.tobytes(),dtype='<f8').reshape(len(packed),373).copy()
    result=dict(integration=values[:,:291],qpos=values[:,291:321],qvel=values[:,321:350],force=values[:,350:373])
    exact(result['integration'][:,1:31],result['qpos'],'packed qpos overlap')
    exact(result['integration'][:,31:60],result['qvel'],'packed qvel overlap')
    return result


def typed_json(payload):
    """Reject truncated/unsupported evidence instead of guessing omitted bytes."""
    if type(payload) is not bytes:raise TypeError('owned JSON bytes required')
    def decode(v,depth=0):
        if depth>12 or not isinstance(v,dict) or v.get('truncated',False) or v.get('unsupported',False):
            raise ValueError('unsupported/truncated typed evidence')
        kind=v.get('type')
        if kind in ('builtins.NoneType','builtins.bool','builtins.int','builtins.str'):
            value=v['value'];expected={'builtins.NoneType':type(None),'builtins.bool':bool,'builtins.int':int,'builtins.str':str}[kind]
            if type(value) is not expected:raise ValueError('typed scalar mismatch')
            return value
        if kind=='builtins.float':return float.fromhex(v['hex'])
        if kind=='builtins.bytes':
            raw=base64.b64decode(v['prefix'],validate=True)
            if type(v['size']) is not int or len(raw)!=v['size'] or hashlib.sha256(raw).hexdigest()!=v['sha256']:
                raise ValueError('typed bytes length/hash mismatch')
            return raw
        if kind=='numpy.ndarray':
            dtype=np.dtype(v['dtype']);shape=v['shape']
            if dtype.str not in ('<f8','<f4','<i8','<i4','|u1') or not isinstance(shape,list) or any(type(n) is not int or n<0 for n in shape):
                raise ValueError('unsupported typed array schema')
            raw=decode(v['data'],depth+1)
            if type(raw) is not bytes or len(raw)!=math.prod(shape)*dtype.itemsize:raise ValueError('array byte count mismatch')
            return np.frombuffer(raw,dtype=dtype).reshape(shape).copy()
        if kind in ('builtins.dict','builtins.list','builtins.tuple'):
            items=v['items']
            if type(v['size']) is not int or len(items)!=v['size']:raise ValueError('container size differs')
            entries=[(decode(k,depth+1),decode(x,depth+1)) for k,x in items]
            if kind=='builtins.dict':
                result={}
                for key,value in entries:
                    if type(key) is not str or key in result:raise ValueError('duplicate/non-string evidence key')
                    result[key]=value
                return result
            if [key for key,_ in entries]!=list(range(len(entries))):raise ValueError('sequence indices differ')
            result=[x for _,x in entries]
            return tuple(result) if kind=='builtins.tuple' else result
        if kind in ('clock_core.CapturedStep','clock_core.WarningLedger'):
            expected={'simulation_time','state','torque','warnings'} if kind.endswith('CapturedStep') else {'counts','lastinfo'}
            if set(v['fields'])!=expected:raise ValueError('dataclass field mismatch')
            return {'__type__':kind,**{k:decode(x,depth+1) for k,x in v['fields'].items()}}
        raise ValueError('unsupported evidence type '+str(kind))
    return decode(json.loads(payload))


def repeated_time(start,count):
    if type(start) is not float or not math.isfinite(start) or start<0:raise ValueError('initial clock')
    values=np.empty(count,dtype=np.float64)
    clock=start
    for i in range(count):clock+=.002;values[i]=clock
    return values


def expected_boundaries(segments):
    names=[];states=[];step_positions=[];offset=0
    for name,z in segments:
        names.append(name+'_initial');states.append(z['initial_integration']);step_positions.append(offset)
        local=0
        for row,control in enumerate(z['global_control']):
            names.append(name+'_precontrol_'+str(int(control)));states.append(z['control_integration_before'][row]);step_positions.append(offset+local)
            local+=int(z['physics_substeps'][row])
        offset+=local
        names.append(name+'_final');states.append(z['final_integration']);step_positions.append(offset)
    return np.asarray(names,dtype='U96'),np.asarray(states,np.float64),np.asarray(step_positions,np.int64)


def compare_case(capture,segments,contract):
    """Vectorized sample comparisons plus every full291 recorded boundary."""
    decoded=unpack373(capture['packed_capture']);n=len(decoded['integration'])
    counts=[int(z['physics_substeps'].sum()) for _,z in segments]
    if n!=sum(counts):raise AssertionError('captured step count differs')
    expected={k:[] for k in ['qpos','qvel','force','command','time','warnings','lastinfo','control','substep','segment','pd','strict']}
    initial=segments[0][1]['initial_integration']
    for index,(name,z) in enumerate(segments):
        steps=counts[index];warning='physics_warning_number' if 'physics_warning_number' in z else 'physics_warning_counts'
        force='physics_actuator_force' if 'physics_actuator_force' in z else 'physics_actuator_torque'
        if index:exact(segments[index-1][1]['final_integration'],z['initial_integration'],'continuous segment boundary')
        exact(z['initial_integration'],z['control_integration_before'][0],'source initial boundary')
        exact(z['initial_integration'][1:31],z['physics_qpos'][0],'source initial qpos')
        exact(z['initial_integration'][31:60],z['physics_qvel'][0],'source initial qvel')
        exact(z['final_integration'][1:31],z['physics_qpos'][-1],'source final qpos')
        exact(z['final_integration'][31:60],z['physics_qvel'][-1],'source final qvel')
        controls=np.repeat(z['global_control'],z['physics_substeps'])
        targets=np.repeat(z['target'],z['physics_substeps'],axis=0)
        previous_q=z['physics_qpos'][:-1,7:];previous_v=z['physics_qvel'][:-1,6:]
        pd=np.minimum(np.maximum(contract['kp']*(targets-previous_q)-contract['kd']*previous_v,-contract['native_effort']),contract['native_effort'])
        expected['qpos'].append(z['physics_qpos'][1:]);expected['qvel'].append(z['physics_qvel'][1:])
        expected['force'].append(z[force]);expected['command'].append(z['physics_torque']);expected['pd'].append(pd)
        expected['time'].append(z['physics_time'][1:]);expected['warnings'].append(z[warning][1:]);expected['lastinfo'].append(z['physics_warning_lastinfo'][1:])
        expected['control'].append(controls);expected['substep'].append(np.concatenate([np.arange(1,int(v)+1,dtype=np.int64) for v in z['physics_substeps']]))
        expected['segment'].append(np.full(steps,name,dtype='U32'))
        exact(z['physics_time'],z['physics_expected_time'],'source expected clock')
        if int(z['integration_state_spec'])!=8191:raise AssertionError('source full291 spec')
        if z['control_integration_before'].shape!=(len(z['target']),291):raise AssertionError('source boundary shape')
        ends=np.cumsum(z['physics_substeps'])
        exact(z['physics_qpos'][ends],z['qpos'][1:],'source per-control qpos')
        exact(z['physics_qvel'][ends],z['qvel'][1:],'source per-control qvel')
    e={key:np.concatenate(values) for key,values in expected.items() if values}
    for name in ['qpos','qvel','force']:exact(decoded[name],e[name],name)
    for name,key in [('command','commanded_torque'),('time','simulation_time'),('warnings','warning_counts'),('lastinfo','warning_lastinfo'),('control','control'),('substep','substep'),('segment','segment')]:exact(capture[key],e[name],name)
    exact(capture['commanded_torque'],e['pd'],'independent native PD arithmetic')
    exact(decoded['integration'][:,89:112],capture['commanded_torque'],'integration ctrl')
    exact(decoded['integration'][:,0],capture['simulation_time'],'integration clock')
    exact(capture['simulation_time'],repeated_time(float(initial[0]),n),'independent repeated2ms clock')
    if not np.isfinite(decoded['integration']).all() or np.any(decoded['integration'][:,112:]):raise AssertionError('nonfinite integration or external forces')
    names,states,positions=expected_boundaries(segments)
    exact(capture['boundary_name'],names,'boundary names/order')
    exact(capture['boundary_integration'],states,'all recorded full291 boundaries')
    for name,state,position in zip(names,states,positions):
        actual=initial if position==0 else decoded['integration'][position-1]
        exact(state,actual,'boundary vs preceding captured full291 '+name)
    # Reconstruct strict predicates separately from the adapter's assessment.
    q=decoded['qpos'];dq=decoded['qvel'];force=decoded['force'];limits=contract['joint_limits']
    excess=np.maximum(np.maximum(limits[:,0]-q[:,7:],q[:,7:]-limits[:,1]),0.).max(axis=1)
    speeds=(np.abs(dq[:,6:])/contract['native_velocity']).max(axis=1)
    effort=(np.abs(force)/contract['native_effort']).max(axis=1)
    tilt=np.arccos(np.clip(1-2*(q[:,4]**2+q[:,5]**2),-1,1))
    reasons=[]
    for i in range(n):
        values=[]
        if not np.isfinite(q[i]).all() or not np.isfinite(dq[i]).all():values.append('nonfinite_state')
        if abs(float(np.linalg.norm(q[i,3:7]))-1.)>1e-10:values.append('invalid_root_quaternion')
        if excess[i]>1e-6:values.append('native_joint_bound')
        if speeds[i]>1.:values.append('native_joint_speed')
        if not np.isfinite(force[i]).all() or effort[i]>1+1e-9:values.append('native_actuator_effort')
        if q[i,2]<.25 or tilt[i]>1.2:values.append('fall')
        if np.any(capture['warning_counts'][i]):values.append('engine_warning')
        if values:reasons.append({'row':i,'control':int(capture['control'][i]),'substep':int(capture['substep'][i]),'reasons':values})
    return dict(samples=n,boundaries=len(names),all_sample_bytes_exact=True,all_boundary_bytes_exact=True,
                repeated_clock_exact=True,strict_failures=reasons),decoded


def compare_fault(payload,capture_payload,report,trace,case_capture):
    fault=typed_json(payload);returned=typed_json(capture_payload)
    if report['reason']!='STRICT_NATIVE_VIOLATION' or report['stage']!='STEP_VERIFY' or report['detail']!='native_joint_bound':raise AssertionError('expected direct strict fault differs')
    for key in ['attempted','returned','captured']:
        if fault[key]!=3158 or report[key]!=3158:raise AssertionError('fault counter '+key)
    if fault['integration_get_returned'] is not True or fault['field_errors']!={}:raise AssertionError('incomplete fault evidence')
    decoded=unpack373(case_capture['packed_capture']);command=case_capture['commanded_torque'][-1]
    for name,value in [('integration',trace['final_integration']),('qpos',trace['physics_qpos'][-1]),('qvel',trace['physics_qvel'][-1]),('ctrl',command)]:exact(fault[name],value,'fault '+name)
    exact(np.frombuffer(fault['selected_target'],dtype='<f8'),trace['target'][-1],'fault selected target')
    exact(np.frombuffer(fault['selected_command'],dtype='<f8'),command,'fault selected command')
    exact(fault['qfrc_actuator'][6:],decoded['force'][-1],'fault actual generalized actuator force')
    if fault['qfrc_actuator'].shape!=(29,) or np.any(fault['qfrc_actuator'][:6]):raise AssertionError('root actuator force')
    for name,shape in [('qfrc_applied',(29,)),('xfrc_applied',(25,6))]:
        if fault[name].dtype!=np.float64 or fault[name].shape!=shape or np.any(fault[name]):raise AssertionError('fault external force '+name)
    exact(fault['warnings'],case_capture['warning_counts'][-1],'fault warnings')
    exact(fault['lastinfo'],case_capture['warning_lastinfo'][-1],'fault warning lastinfo')
    exact(np.asarray(fault['time'],np.float64),case_capture['simulation_time'][-1],'fault clock')
    if returned['__type__']!='clock_core.CapturedStep' or returned['warnings']['__type__']!='clock_core.WarningLedger':raise AssertionError('returned capture type')
    if returned['state']!=case_capture['packed_capture'][-1].tobytes() or returned['torque']!=command.tobytes():raise AssertionError('fault returned raw capture differs')
    exact(np.asarray(returned['simulation_time'],np.float64),case_capture['simulation_time'][-1],'returned fault time')
    exact(np.asarray(returned['warnings']['counts'],np.int32),case_capture['warning_counts'][-1],'returned warnings')
    exact(np.asarray(returned['warnings']['lastinfo'],np.int32),case_capture['warning_lastinfo'][-1],'returned lastinfo')
    return dict(full_fault_and_returned_capture_exact=True,expected_physical_failure=True)
