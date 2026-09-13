"""Independent differential and recoverability checks for reference leg IK."""
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import ROOT,MODEL,load_motion
from gear_sonic.utils.g1_true23_intent_retarget import Native23LegIK,kinematics,torso_matched_base


@pytest.fixture(scope="module")
def setup():
    model=mujoco.MjModel.from_xml_path(str(ROOT.parent/"GR00T-WholeBodyControl"/MODEL))
    motion,_,_=load_motion("walk002")
    qpos=np.r_[motion["body_pos_w"][0,0],motion["body_quat_w"][0,0],motion["joint_pos"][0]]
    return model,qpos


def test_leg_jacobian_matches_finite_differences(setup):
    model,qpos=setup
    solver=Native23LegIK(model)
    for side in range(2):
        indices=np.arange(side*6,side*6+6)
        q=qpos[7+indices]+np.array([.12,.07,-.05,.1,.03,-.02])
        position=np.array([.1,(-1)**side*.12,.1])
        rotation=Rotation.from_euler("xyz",[.1,-.15,.2]).as_matrix()
        args=(side,qpos,position,rotation,qpos[7+indices])
        _,jac=solver.residual_jacobian(q,*args)
        numeric=np.empty_like(jac)
        for j in range(6):
            delta=np.zeros(6);delta[j]=1e-6
            numeric[:,j]=(solver.residual_jacobian(q+delta,*args)[0]-solver.residual_jacobian(q-delta,*args)[0])/2e-6
        np.testing.assert_allclose(jac,numeric,atol=2e-8,rtol=1e-6)


def test_recovers_reachable_feet_without_changing_other_axes(setup):
    model,qpos=setup
    data=mujoco.MjData(model)
    solver=Native23LegIK(model,posture_weight=0.)
    wanted=qpos.copy()
    wanted[7:19]+=np.tile([-.1,.07,.05,.12,.04,-.03],2)
    data.qpos[:]=wanted
    kinematics(model,data)
    positions=data.xpos[solver.bodies].copy()
    rotations=data.xmat[solver.bodies].reshape(2,3,3).copy()
    result,errors,angles,_=solver.solve(qpos,positions,rotations)
    assert errors.max()<1e-7
    assert angles.max()<1e-6
    np.testing.assert_array_equal(result[:7],qpos[:7])
    np.testing.assert_array_equal(result[19:],qpos[19:])
    assert np.all(result[7:]>=model.jnt_range[1:,0])
    assert np.all(result[7:]<=model.jnt_range[1:,1])


def test_torso_base_preserves_requested_world_pose(setup):
    model,seed=setup
    data=mujoco.MjData(model)
    position=np.array([.2,-.1,.81])
    rotation=Rotation.from_euler("xyz",[.2,-.25,.43]).as_matrix()
    seed=seed.copy();seed[19]=.3
    result=torso_matched_base(model,data,seed,position,rotation)
    data.qpos[:]=result
    mujoco.mj_kinematics(model,data)
    torso=model.body("torso_link").id
    np.testing.assert_allclose(data.xpos[torso],position,atol=1e-12)
    np.testing.assert_allclose(data.xmat[torso].reshape(3,3),rotation,atol=1e-12)
    np.testing.assert_array_equal(result[7:],seed[7:])


def test_bounded_height_relief_reduces_unreachable_foot_error(setup):
    model,qpos=setup
    solver=Native23LegIK(model,max_nfev=80)
    data=mujoco.MjData(model)
    data.qpos[:]=qpos
    kinematics(model,data)
    positions=data.xpos[solver.bodies].copy()
    rotations=data.xmat[solver.bodies].reshape(2,3,3).copy()
    raised=qpos.copy();raised[2]+=.08
    _,fixed_errors,_,_=solver.solve(raised,positions,rotations)
    result,errors,_,_=solver.solve_with_height(raised,positions,rotations)
    assert errors.max()<fixed_errors.max()*.1
    assert 0<raised[2]-result[2]<=.060000001
    np.testing.assert_array_equal(result[3:7],raised[3:7])
    np.testing.assert_array_equal(result[19:],raised[19:])


def test_coupled_leg_step_box_keeps_joint_specific_bounds_and_fixed_axes(setup):
    model,qpos=setup
    solver=Native23LegIK(model,posture_weight=.002,max_nfev=80)
    data=mujoco.MjData(model)
    desired=qpos.copy()
    desired[7:19]+=np.tile([-.2,.1,.1,.35,-.1,.1],2)
    data.qpos[:]=desired
    kinematics(model,data)
    positions=data.xpos[solver.bodies].copy()
    rotations=data.xmat[solver.bodies].reshape(2,3,3).copy()
    previous=qpos.copy()
    current=qpos.copy();current[7:19]=model.jnt_range[1:13,1]-.01
    step=np.linspace(.004,.02,12)
    fitted,errors,_,_=solver.solve_with_height(current,positions,rotations,
                                               posture_qpos=previous,previous_qpos=previous,max_step_rad=step)
    assert np.all(np.abs(fitted[7:19]-previous[7:19])<=step+1e-12)
    assert np.all(fitted[7:19]>=model.jnt_range[1:13,0])
    assert np.all(fitted[7:19]<=model.jnt_range[1:13,1])
    assert errors.max()>.01  # Goal loss stays visible when the requested step is unreachable.
    np.testing.assert_array_equal(fitted[:2],current[:2])
    np.testing.assert_array_equal(fitted[3:7],current[3:7])
    np.testing.assert_array_equal(fitted[19:],current[19:])
    assert abs(fitted[2]-current[2])<=.060000001
    assert solver.last_solution_info['success']
    assert len(solver.last_solution_info['attempts'])==2
    assert solver.last_solution_info['cost']==min(a['cost'] for a in solver.last_solution_info['attempts'])
    assert solver.last_solution_info['total_nfev']>=solver.last_solution_info['nfev']


@pytest.mark.parametrize('step',(0.,-.1,np.nan,np.inf,[.1,.2]))
def test_coupled_leg_rejects_invalid_step_box(setup,step):
    model,qpos=setup
    solver=Native23LegIK(model)
    with pytest.raises(ValueError,match='max_step_rad'):
        solver.solve_with_height(qpos,np.zeros((2,3)),np.tile(np.eye(3),(2,1,1)),
                                  previous_qpos=qpos,max_step_rad=step)


def test_coupled_leg_requires_previous_and_nonempty_physical_intersection(setup):
    model,qpos=setup
    solver=Native23LegIK(model)
    args=(qpos,np.zeros((2,3)),np.tile(np.eye(3),(2,1,1)))
    with pytest.raises(ValueError,match='supplied together'):
        solver.solve_with_height(*args,previous_qpos=qpos)
    with pytest.raises(ValueError,match='supplied together'):
        solver.solve_with_height(*args,max_step_rad=.1)
    previous=qpos.copy();previous[7:19]=model.jnt_range[1:13,1]+.2
    with pytest.raises(ValueError,match='no interior'):
        solver.solve_with_height(*args,previous_qpos=previous,max_step_rad=.01)


def test_coupled_height_jacobian_matches_finite_differences(setup,monkeypatch):
    import gear_sonic.utils.g1_true23_intent_retarget as module
    model,qpos=setup
    solver=Native23LegIK(model)
    qpos=qpos.copy();qpos[3:7]=Rotation.from_euler('xyz',[.12,-.13,.21]).as_quat()[[3,0,1,2]]
    qpos[7:19]+=np.tile([.1,.02,-.02,.12,.02,-.02],2)
    real_least_squares=module.least_squares
    observed=[]
    def checked(fun,x0,*,jac,**kwargs):
        analytic=jac(x0)
        numerical=np.column_stack([(fun(x0+d)-fun(x0-d))/2e-6 for d in np.eye(13)*1e-6])
        np.testing.assert_allclose(analytic,numerical,atol=2e-8,rtol=1e-6)
        observed.append(True)
        return real_least_squares(fun,x0,jac=jac,**kwargs)
    monkeypatch.setattr(module,'least_squares',checked)
    solver.solve_with_height(qpos,np.array([[.1,.15,.04],[.05,-.13,.05]]),
                               np.tile(Rotation.from_euler('xyz',[.1,.2,-.1]).as_matrix(),(2,1,1)),
                               previous_qpos=qpos,max_step_rad=np.linspace(.1,.2,12))
    assert observed==[True]


def test_coupled_solver_budget_failure_is_visible(setup):
    model,qpos=setup
    solver=Native23LegIK(model,max_nfev=1)
    result,errors,_,counts=solver.solve_with_height(qpos,np.full((2,3),4.),np.tile(np.eye(3),(2,1,1)),
                                                    previous_qpos=qpos,max_step_rad=.01)
    assert not solver.last_solution_info['success']
    assert solver.last_solution_info['status']==0
    assert np.all(counts==1)
    assert errors.min()>1.
    assert np.max(np.abs(result[7:19]-qpos[7:19]))<=.01
