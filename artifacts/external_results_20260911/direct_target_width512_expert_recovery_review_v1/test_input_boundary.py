"""Independent synthetic extraction cases; never load task data or native code."""
import sys
import unittest
from pathlib import Path
import numpy as np

SOURCE = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_width512_expert_recovery_v1/source_draft_v1')
sys.path.insert(0, str(SOURCE))
from recovery_inputs import extract, named
from recovery_contract import protocol, budgets


def fixture():
    n = 260
    clock = [0.0]
    for _ in range(10*n):
        clock.append(clock[-1] + .002)
    clock = np.asarray(clock, np.float64)
    q = np.zeros((n+1,30), np.float64)
    q[:,0] = np.arange(n+1)*.0001
    q[:,2] = .75
    q[:,3] = 1.
    dq = np.zeros((n+1,29), np.float64)
    dq[:,0] = np.arange(n+1)*.001
    full = np.zeros((n,291), np.float64)
    full[:,0] = clock[np.arange(n)*10]
    full[:,1:31] = q[:-1]
    full[:,31:60] = dq[:-1]
    full[:,60:] = np.arange(n)[:,None] + np.arange(231)[None,:]/512
    action = np.zeros((n,23),np.float32)
    action[250] = np.arange(23,dtype=np.float32)/64
    previous = np.concatenate((np.zeros((1,23),np.float32),action[:-1]))
    target = np.zeros((n,23),np.float64)
    target[250] = action[250].astype(np.float64)/8
    history = (np.arange(n)[:,None]*1024 + np.arange(300)[None,:]).astype(np.float32)
    physics_q = np.repeat(q[:-1],10,axis=0)
    physics_q = np.concatenate((physics_q,q[-1:]))
    physics_dq = np.repeat(dq[:-1],10,axis=0)
    physics_dq = np.concatenate((physics_dq,dq[-1:]))
    warnings = np.zeros((10*n+1,8),np.int32)
    warning_info = np.zeros_like(warnings)
    # Deliberately different synthetic later evidence proves boundary selection.
    warning_info[2510] = np.arange(8,dtype=np.int32)+17
    warning_info[-1] = 99
    trace = dict(target=target,source_frame=np.arange(n,dtype=np.int64)+11,
        global_control=np.arange(n,dtype=np.int64),controller_mode=np.r_[np.zeros(250,np.int64),np.ones(n-250,np.int64)],
        joint_error=np.zeros((n,23),np.float64),root_error=np.zeros((n,3),np.float64),
        physics_substeps=np.full(n,10,np.int64),control_integration_before=full,
        control_history_before=history,history=history.copy(),
        control_previous_action_before=previous,previous_action=previous.copy(),action=action,
        qpos=q,qvel=dq,physics_qpos=physics_q,physics_qvel=physics_dq,
        physics_time=clock,physics_expected_time=clock.copy(),
        physics_warning_counts=warnings,physics_warning_lastinfo=warning_info,
        physics_torque=np.zeros((10*n,23),np.float64),physics_actuator_torque=np.zeros((10*n,23),np.float64),
        initial_integration=full[0].copy())
    for key in ('range_excess','velocity_ratio','effort_ratio','clock_error'):
        trace[key]=np.zeros(10*n,np.float64)
    # Fields used by the adapter's copied-prefix metadata, when retained.
    trace['delta']=np.zeros((n,23),np.float64)
    trace['delta'][250]=target[250]
    contract=dict(kp=np.full(23,80.),training_effort=np.full(23,40.),default_q=np.zeros(23))
    return trace,contract


class InputBoundary(unittest.TestCase):
    def test_actual_boundary_and_warmstart_are_copied(self):
        trace,c=fixture();snapshot,prefix=extract(trace,c)
        self.assertEqual(snapshot['integration'].tobytes(),trace['control_integration_before'][251].tobytes())
        self.assertEqual(snapshot['warning_lastinfo'].tobytes(),trace['physics_warning_lastinfo'][2510].tobytes())
        self.assertEqual(len(prefix['target']),251)
        self.assertEqual(len(prefix['qpos']),252)
        self.assertEqual(len(prefix['physics_torque']),2510)
        self.assertEqual(len(prefix['physics_time']),2511)
        self.assertEqual(prefix['controller_mode'][-1],1)
        self.assertEqual(snapshot['previous_action'].tobytes(),trace['action'][250].tobytes())
        self.assertEqual(snapshot['history_flat'].tobytes(),trace['history'][251].tobytes())
        self.assertNotEqual(snapshot['history_flat'].tobytes(),trace['history'][250].tobytes())
        prefix['qpos'][-1,0]=999
        snapshot['integration'][-1]=999
        self.assertNotEqual(trace['qpos'][251,0],999)
        self.assertNotEqual(trace['control_integration_before'][251,-1],999)

    def test_outgoing_raw_substitution_rejected(self):
        trace,c=fixture()
        trace['control_previous_action_before'][251,0] = .125
        trace['previous_action'][251,0] = .125
        trace['action'][250,0] = .125
        with self.assertRaisesRegex(ValueError,'applied target250 inverse'):
            extract(trace,c)

    def test_clock_one_ulp_change_rejected(self):
        trace,c=fixture()
        trace['physics_time'][2510]=np.nextafter(trace['physics_time'][2510],np.inf)
        with self.assertRaisesRegex(ValueError,'accumulated clock'):
            extract(trace,c)

    def test_named_history_offsets_and_dtype(self):
        a=np.arange(300,dtype=np.float32);h=named(a)
        for key,lo,hi,shape in [('actions',0,92,(4,23)),('base_ang_vel',92,104,(4,3)),
            ('dof_pos',104,196,(4,23)),('dof_vel',196,288,(4,23)),('projected_gravity',288,300,(4,3))]:
            self.assertEqual(h[key].shape,shape)
            self.assertEqual(h[key].tobytes(),a[lo:hi].tobytes())
        with self.assertRaises(ValueError):named(a.astype(np.float64))

    def test_planning_tail_and_declared_units(self):
        p=protocol();b=budgets()
        self.assertEqual(p['plan_controls'],list(range(251,1269,5)))
        self.assertEqual(len(p['plan_controls']),204)
        self.assertEqual(p['plan_controls'][-1],1266)
        self.assertEqual(p['final_MPC_commit'],3)
        self.assertEqual(p['requested_branch_controls'],1318)
        self.assertEqual(b['native_live_step'],(15680,15680))
        self.assertEqual(b['native_private_step'],(200180,200180))
        self.assertEqual(b['batch_fd_step'][1],150552000)
        self.assertEqual(b['batch_line_step'][1],20933100)
        self.assertEqual(b['BFM_actor'],(6670,6670))
        self.assertEqual(b['BFM_backward'],(6670,53360))


if __name__=='__main__':unittest.main()
