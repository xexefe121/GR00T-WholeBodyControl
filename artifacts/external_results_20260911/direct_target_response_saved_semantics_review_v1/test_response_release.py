"""Synthetic metadata regressions; never read task arrays or run task models."""
import ast
import copy
import unittest
from saved_common import BASE,NEW,SUBJECTS,Checks
from release_checks import binding_identity,warm_identity,intent_identity

def warm():
    return dict(condition='causal',features=1323,context_features=323,ordinary_start_step=68000,
        ordinary_final_step=71000,additional_updates=3000,optimizer_start_step=3000,optimizer_step=6000,
        fresh_optimizer=False,fixed_full_state_coefficient=1.8188207859141674,context_and_normalization_reused=True,
        response_weight_rule='mean_six_teacher_group_energies_over_group_energy',response_group_weights=[1.,2.,3.,4.,5.,6.],
        source_checkpoint_sha256='10d238198e2d7826c35eaa9bd4a13ff2e05050a7f601c1f212b585c4a90eaefd')

class ReleaseTests(unittest.TestCase):
    def test_sixteen_roles_exact(self):
        self.assertEqual(SUBJECTS,('fit_report','checkpoint','head','normalization','training_manifest','training_request','export_manifest','coefficient','source_checkpoint','full_state_generation_request','full_state_generation_report','full_state_data_audit','full_state_data_owner','shared_manifest','context_alignment','energy_source'))

    def test_warm_metadata(self):warm_identity(warm(),dict(rule='Emean/Eg',group_weights=[1.,2.,3.,4.,5.,6.]),Checks())

    def test_wrong_warm_identity(self):
        for key,value in [('fresh_optimizer',True),('optimizer_start_step',0),('optimizer_step',3000),('ordinary_final_step',68000),('additional_updates',6000),('context_and_normalization_reused',False),('condition','blinded'),('fixed_full_state_coefficient',1.),('response_weight_rule','Emean/Eg'),('source_checkpoint_sha256','0'*64)]:
            fit=warm();fit[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):warm_identity(fit,dict(rule='Emean/Eg',group_weights=[1.,2.,3.,4.,5.,6.]),Checks())

    def test_energy_rule_and_group_order(self):
        for energy in [dict(rule=warm()['response_weight_rule'],group_weights=[1.,2.,3.,4.,5.,6.]),dict(rule='Emean/Eg',group_weights=[6.,5.,4.,3.,2.,1.])]:
            with self.assertRaises(AssertionError):warm_identity(warm(),energy,Checks())

    def test_failed_intent_is_preserved(self):
        intent=dict(kind='independent_recorded_state_intent_inspection',requested_controls=1569,global_start=0,
            recorded_controls=271,intended_segment_completed=False,dynamics_executed=False,hardware_authorized=False,
            full_lifecycle_source_intent_pass=False,requested_segment_quiet_pass=False)
        report=dict(attempted_controls=271,full_segment_completed=False)
        intent_identity(intent,report,Checks())
        for key,value in [('recorded_controls',270),('requested_controls',271),('intended_segment_completed',True),('dynamics_executed',True),('requested_segment_quiet_pass',None)]:
            bad=copy.deepcopy(intent);bad[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):intent_identity(bad,report,Checks())

    def test_no_segment_or_map_math_delta(self):
        old=NEW/'direct_target_context_saved_semantics_review_v1'
        for name in ('audit_saved.py','context_math.py','fixed_maps.py','test_saved.py'):
            self.assertEqual((BASE/name).read_bytes(),(old/name).read_bytes(),name)

    def test_common_only_constant_delta(self):
        old=NEW/'direct_target_context_saved_semantics_review_v1/saved_common.py'
        keep=lambda p:[ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        self.assertEqual(keep(old),keep(BASE/'saved_common.py'))

if __name__=='__main__':unittest.main()
