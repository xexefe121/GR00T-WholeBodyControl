"""Narrow81000 release regressions; no task arrays or model execution."""
import ast,copy,unittest
from saved_common import BASE,NEW,Checks
from release_checks import binding_identity,warm_identity
from test_response_release import warm

class WidthReleaseTests(unittest.TestCase):
    def test_exact_width_binding_and_old256_rejected(self):
        binding=dict(context_condition='causal',architecture=[1323,512,512,23],ordinary_final_step=81000,
            release_kind='ordinary81000_width512_warm_balanced_context_same_weight_fp64_export')
        binding_identity(binding,Checks())
        for key,value in [('architecture',[1323,256,256,23]),('ordinary_final_step',71000),('release_kind','ordinary71000_warm_balanced_context_same_weight_fp64_export')]:
            bad=copy.deepcopy(binding);bad[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):binding_identity(bad,Checks())

    def test_expansion_metadata_corruptions(self):
        energy=dict(rule='Emean/Eg',group_weights=[1.,2.,3.,4.,5.,6.])
        for key,value in [('architecture',[1323,256,256,23]),('hidden_width',256),('expansion_seed',20260911),
            ('training_first_layer_execution','split_contiguous_1000_plus_323'),('export_first_layer_execution','split_float64'),
            ('ordinary_start_step',68000),('optimizer_start_step',3000),('optimizer_step',6000),('additional_updates',3000)]:
            fit=warm();fit[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):warm_identity(fit,energy,Checks())

    def test_future_request_requires_width_and_endpoint(self):
        text=(BASE/'prepare_request.py').read_text()
        self.assertIn("binding['ordinary_final_step']==81000 and binding['architecture']==[1323,512,512,23]",text)
        self.assertIn('one_pure_saved_causal81000_width512_runtime_semantics_and_fixed_maps',text)
        self.assertIn('original_requested_controls=1569,conditional_continuous_hold_controls=250',text)

    def test_durable_helper_functions_unchanged(self):
        old=NEW/'direct_target_response_saved_semantics_review_v1'
        functions=lambda p:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        self.assertEqual(functions(BASE/'prepare_audit_stage.py'),functions(old/'prepare_audit_stage.py'))
        self.assertEqual((BASE/'verify_completion.py').read_bytes(),(old/'verify_completion.py').read_bytes())

    def test_same_controller_features_and_history_sources(self):
        old=NEW/'direct_target_causal_response_evaluation_v1/source_draft_v1'
        new=NEW/'direct_target_causal_width512_evaluation_v1/source_draft_v1'
        for name in ('direct_features.py','causal_features.py','direct_runtime.py','evaluate_direct_target_student.py','proposal_evidence.py','head_activation_witness.py'):
            self.assertEqual((old/name).read_bytes(),(new/name).read_bytes(),name)

if __name__=='__main__':unittest.main()
