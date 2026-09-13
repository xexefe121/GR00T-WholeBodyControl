from types import SimpleNamespace
import json

import mujoco
import numpy as np
import pytest

from gear_sonic.utils.g1_true23_native_preview_guard import NativePreviewGuard
from gear_sonic.utils.g1_true23_sim_preview import (
    SimulatorPreviewRejected, publish_checked, stop_simulator,
    terminal_standing, write_diagnostics)


@pytest.fixture
def guard():
    bodies=''.join(f'<body pos="{i*.1} 0 1"><joint name="j{i}" range="-1 1"/>'
                   '<geom type="sphere" size=".02" mass="1"/></body>' for i in range(23))
    motors=''.join(f'<motor joint="j{i}"/>' for i in range(23))
    model=mujoco.MjModel.from_xml_string('<mujoco><compiler angle="radian"/>'
        '<option timestep=".002"/><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/>'
        f'{bodies}</body></worldbody><actuator>{motors}</actuator></mujoco>')
    contract=dict(kp=np.ones(23)*20,kd=np.ones(23),native_effort=np.ones(23)*10,
                  joint_limits=np.tile([-1.,1.],(23,1)))
    return NativePreviewGuard(model,contract,delay_substeps=6)


def run(guard, errors, boundary='lower'):
    errors=iter(errors)
    def preview(target):
        low=np.full(23,-.5);high=low.copy()
        (low if boundary=='lower' else high)[19]=next(errors)
        return low,high
    guard._preview=preview
    q=np.zeros(30);q[3]=1
    return guard.apply(q,np.zeros(29),np.zeros(23),np.ones(23)*.1)


@pytest.mark.parametrize('errors,calls,acceptable',[
    ([-.01],1,True),([.1,.1,.1],3,False),
    ([.7,.6,.5,.4,.3,.2,.1],7,False),([.1,.01,-.01],3,True),
    ([.00005,.00005,.00005],3,False)])
@pytest.mark.parametrize('boundary',['lower','upper'])
def test_bounded_search_and_owned_evidence(guard,errors,calls,acceptable,boundary):
    target,status=run(guard,errors,boundary)
    evidence=guard.last_diagnostic
    assert status['preview_calls']==calls
    assert status['predicted_limits_satisfied']==acceptable
    assert status['horizon_seconds']==.032
    assert evidence['limiting_joint_index']==19
    assert evidence['limiting_boundary']==boundary
    assert max(evidence['predicted_lower_excess_rad'][19],evidence['predicted_upper_excess_rad'][19])==errors[-1]
    before=evidence['candidate_target'].copy();target[:]=99
    np.testing.assert_array_equal(evidence['candidate_target'],before)


@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf])
def test_nonfinite_preview_fails(guard,value):
    with pytest.raises(ValueError,match='Nonfinite'):
        run(guard,[value])
    assert guard.last_diagnostic is None


@pytest.mark.parametrize('control',[0,37])
def test_rejection_prevents_publication_and_commit(control):
    calls=[];commits=[];records=[]
    command=SimpleNamespace(targets=np.zeros(23),status={'native_preview_guard':{'predicted_limits_satisfied':False}})
    def publish():
        calls.append(True);commits.append(command.targets)
        return [1,1]
    with pytest.raises(SimulatorPreviewRejected):
        publish_checked(command,{'requested_target':np.zeros(23)},dict(control=control),
                        'strict',records,publish)
    assert not calls and not commits
    assert not records[0]['published'] and records[0]['application_time_ns'] is None


@pytest.mark.parametrize('policy,acceptable',[('strict',True),('diagnostic-only',False)])
def test_allowed_publication(policy,acceptable):
    records=[];calls=[]
    command=SimpleNamespace(targets=np.zeros(23),status={'native_preview_guard':{'predicted_limits_satisfied':acceptable}})
    assert publish_checked(command,{},dict(control=2),policy,records,lambda:calls.append(1) or [1,1])==[1,1]
    assert calls==[1] and records[0]['published']


def test_partial_channel_publication_remains_visible():
    records=[]
    command=SimpleNamespace(targets=np.zeros(23),status={'native_preview_guard':{'predicted_limits_satisfied':True}})
    publish_checked(command,{},dict(control=2),'strict',records,lambda:[1,0])
    assert records[0]['published'] and not records[0]['all_channels_accepted']


def test_nonfinite_command_cannot_be_published_with_safe_status():
    records=[];calls=[]
    command=SimpleNamespace(targets=np.full(23,np.nan),status={'native_preview_guard':{'predicted_limits_satisfied':True}})
    with pytest.raises(SimulatorPreviewRejected):
        publish_checked(command,{},dict(control=2),'strict',records,lambda:calls.append(1))
    assert not calls


def test_stop_precedes_signal_and_serialization(tmp_path):
    order=[]
    bridge=SimpleNamespace(address=1,lib=SimpleNamespace(clock_stop=lambda address:order.append('native_stop')))
    stamp=stop_simulator(bridge,SimpleNamespace(set=lambda:order.append('signal')))
    assert order==['native_stop','signal'] and stamp>0
    records=[dict(control=0,published=True),dict(control=1,published=False)]
    timing=np.zeros((10,7),np.int64);timing[:,0]=np.arange(1,11)*2_000_000
    path=tmp_path/'diagnostics.jsonl';write_diagnostics(path,records,timing,0)
    saved=[json.loads(line) for line in path.read_text().splitlines()]
    assert saved[0]['application_physics_step']==0
    assert saved[1]['application_time_ns'] is None


def test_unreached_terminal_hold_is_not_trailing_quiet():
    final,continuous=terminal_standing({'passed':True},{'passed':True},False)
    assert final['status']==continuous['status']=='not_reached'
    assert not final['passed'] and not continuous['passed']
