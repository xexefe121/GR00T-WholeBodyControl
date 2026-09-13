import copy
import unittest
import numpy as np
from compare_first_target import compare


def fixture():
    snapshot=dict(qpos=np.arange(30,dtype=np.float64),qvel=np.arange(29,dtype=np.float64),
                  previous_action=np.arange(23,dtype=np.float32),history_flat=np.arange(300,dtype=np.float32),
                  integration=np.arange(291,dtype=np.float64))
    target=np.arange(23,dtype=np.float64)
    first=dict(global_control=np.asarray(251),proposal_before_preview=np.asarray(True),target=target+1,
               actual_state=np.r_[snapshot['qpos'],snapshot['qvel']],incoming_prior=snapshot['previous_action'].copy(),
               incoming_history=snapshot['history_flat'].copy())
    rows=dict(control=np.asarray([251]),qpos=snapshot['qpos'][None],qvel=snapshot['qvel'][None],
              previous_action=snapshot['previous_action'][None],history=snapshot['history_flat'][None],target=target[None])
    maps=dict(control=np.asarray([251]),head_target=target[None],map_target=(target+2)[None])
    branch=dict(global_control=np.asarray([],dtype=np.int64))
    return snapshot,first,rows,maps,branch


class TargetComparisonTests(unittest.TestCase):
    def test_saved_proposal_does_not_claim_execution_or_qualification(self):
        result=compare(*fixture())
        self.assertEqual(result['fresh_minus_student']['rmse_rad'],1.)
        self.assertEqual(result['fresh_minus_fixed_map']['difference_by_joint_rad'],[-1.]*23)
        self.assertFalse(result['fresh_target_was_issued'])
        self.assertFalse(result['labels_admissible'])
        self.assertFalse(result['recovery_qualification_inferred'])

    def test_partial_native_control_is_not_complete(self):
        s,f,r,m,b=fixture()
        b.update(global_control=np.asarray([251]),target=f['target'][None],
                 control_integration_before=s['integration'][None],physics_substeps=np.asarray([6]))
        result=compare(s,f,r,m,b)
        self.assertTrue(result['fresh_target_was_issued'])
        self.assertEqual(result['first_control_native_steps'],6)
        self.assertFalse(result['first_control_completed'])

    def test_wrong_context_and_duplicate_control_rejected(self):
        values=fixture()
        wrong=copy.deepcopy(values)
        wrong[1]['incoming_history'][1]+=1
        with self.assertRaises(ValueError):compare(*wrong)
        wrong=copy.deepcopy(values)
        wrong[2]['control']=np.asarray([251,251])
        with self.assertRaises(ValueError):compare(*wrong)


if __name__=='__main__':unittest.main()
