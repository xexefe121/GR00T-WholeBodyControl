"""Pure saved-prefix extraction. No native, planner, or inference imports."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from recovery_contract import START, HISTORY_SHAPES, SOURCE_TRACE_SHA, SEMANTICS_SHA, SEMANTICS_OWNER_SHA

CONTROL = ('target','source_frame','global_control','controller_mode','joint_error','root_error','physics_substeps',
           'control_integration_before','control_history_before','control_previous_action_before','previous_action','action')
STEP_STATES = ('physics_qpos','physics_qvel','physics_time','physics_expected_time','physics_warning_counts','physics_warning_lastinfo')
STEP_OUTPUTS = ('physics_torque','physics_actuator_torque','range_excess','velocity_ratio','effort_ratio','clock_error')


def exact(a,b,name):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():
        raise ValueError('Saved byte identity differs: '+name)


def named(flat):
    flat=np.asarray(flat)
    if flat.shape!=(300,) or flat.dtype!=np.float32 or not np.isfinite(flat).all():
        raise ValueError('Incoming named history requires finite300float32')
    result={};offset=0
    for key,shape in HISTORY_SHAPES.items():
        size=int(np.prod(shape));result[key]=flat[offset:offset+size].copy().reshape(shape);offset+=size
    assert offset==300
    return result


def extract(trace,contract):
    for key,shape,dtype in [('control_integration_before',(291,),np.float64),('control_history_before',(300,),np.float32),
                          ('control_previous_action_before',(23,),np.float32),('qpos',(30,),np.float64),('qvel',(29,),np.float64)]:
        a=trace[key]
        if len(a)<=START or a.shape[1:]!=shape or a.dtype!=dtype or not np.isfinite(a[:START+1]).all():
            raise ValueError('Selected precontrol schema: '+key)
    exact(trace['global_control'][:START+1],np.arange(START+1,dtype=np.int64),'absolute controls')
    exact(trace['source_frame'][:START+1],np.arange(START+1,dtype=np.int64)+11,'original source clock')
    exact(trace['physics_substeps'][:START],np.full(START,10,np.int64),'all251 prefix controls complete')
    exact(trace['controller_mode'][:250],np.zeros(250,np.int64),'original250BFM')
    if int(trace['controller_mode'][250])!=1:raise ValueError('Actual learned action250 missing')
    exact(trace['previous_action'][:START+1],trace['control_previous_action_before'][:START+1],'incoming prior aliases')
    exact(trace['history'][:START+1],trace['control_history_before'][:START+1],'incoming history aliases')
    exact(trace['control_previous_action_before'][1:START+1],trace['action'][:START],'actual outgoing prior chain')
    kp,effort,default=[np.asarray(contract[k],np.float64) for k in ['kp','training_effort','default_q']]
    inverse=((trace['target'][250]-default)*kp/(.25*effort)).astype(np.float32)
    exact(inverse,trace['control_previous_action_before'][START],'applied target250 inverse')
    elapsed=0.
    for i in range(START*10+1):
        if i:elapsed+=.002
        if elapsed!=float(trace['physics_expected_time'][i]) or elapsed!=float(trace['physics_time'][i]):
            raise ValueError('Original accumulated clock changed at '+str(i))
    integration=trace['control_integration_before'][START].copy()
    if float(integration[0])!=elapsed:raise ValueError('Selected full291 clock differs')
    exact(integration[1:31],trace['qpos'][START],'selected integration/qpos')
    exact(integration[31:60],trace['qvel'][START],'selected integration/qvel')
    exact(trace['qpos'][START],trace['physics_qpos'][START*10],'selected boundary qpos')
    exact(trace['qvel'][START],trace['physics_qvel'][START*10],'selected boundary qvel')
    buffers=named(trace['control_history_before'][START])
    snapshot=dict(integration=integration,integration_state_spec=np.asarray(8191,np.int64),
        qpos=trace['qpos'][START].copy(),qvel=trace['qvel'][START].copy(),
        history_flat=trace['control_history_before'][START].copy(),previous_action=inverse.copy(),
        warning_counts=trace['physics_warning_counts'][START*10].copy(),
        warning_lastinfo=trace['physics_warning_lastinfo'][START*10].copy(),
        recorded_controls=np.asarray(START,np.int64),expected_time=np.asarray(elapsed,np.float64),actual_time=np.asarray(elapsed,np.float64),
        **{'history_'+k:v.copy() for k,v in buffers.items()})
    prefix={key:trace[key][:START].copy() for key in (*CONTROL,'history','delta')}
    for key in ['qpos','qvel']:prefix[key]=trace[key][:START+1].copy()
    for key in STEP_STATES:prefix[key]=trace[key][:START*10+1].copy()
    for key in STEP_OUTPUTS:prefix[key]=trace[key][:START*10].copy()
    prefix.update(initial_integration=trace['initial_integration'].copy(),final_integration=integration.copy(),
                  final_previous_action=inverse.copy(),final_recorded_controls=np.asarray(START,np.int64))
    for key,shape in HISTORY_SHAPES.items():
        prefix['control_history_before_'+key]=np.stack([named(v)[key] for v in trace['control_history_before'][:START]])
        prefix['final_history_'+key]=buffers[key].copy()
    return snapshot,prefix


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);args=parser.parse_args()
    request=json.loads(args.request.read_text())
    if request.get('root_selected_saved_input_preparation') is not True:raise ValueError('Actual saved input preparation not selected')
    roles=request['subjects']
    for key,entry in roles.items():
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Changed saved subject: '+key)
    if roles['trace']['sha256']!=SOURCE_TRACE_SHA or roles['semantics']['sha256']!=SEMANTICS_SHA or roles['semantics_owner']['sha256']!=SEMANTICS_OWNER_SHA:
        raise ValueError('Wrong prespecified width trace/qualification')
    with np.load(roles['trace']['path'],allow_pickle=False) as a:trace={k:a[k].copy() for k in a.files}
    snapshot,prefix=extract(trace,json.loads(Path(roles['contract']['path']).read_text()))
    output=Path(request['output']);output.mkdir(exist_ok=False)
    np.savez(output/'precontrol251.npz',**snapshot);np.savez(output/'actual_prefix251.npz',**prefix)
    for entry in roles.values():assert sha(entry['path'])==entry['sha256']
    result=dict(passed=True,source_trace_sha256=SOURCE_TRACE_SHA,selected_control=START,physics_steps_copied=2510,
        controls_copied=251,learned_prefix_controls=1,initial_history_fields=list(HISTORY_SHAPES),
        request_sha256=sha(args.request),input_sha256={v['path']:v['sha256'] for v in roles.values()},
        outputs={p.name:sha(p) for p in output.glob('*.npz')},model_calls=0,native_steps=0,replans=0)
    with (output/'selection_receipt.json').open('x') as f:json.dump(result,f,indent=2)


if __name__=='__main__':main()
