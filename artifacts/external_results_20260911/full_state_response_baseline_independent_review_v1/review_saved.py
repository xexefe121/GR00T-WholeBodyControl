"""Independent source/receipt review and saved-summary arithmetic. No forwards."""
import hashlib,json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;NEW=BASE.parent;OLD=NEW/'full_state_response_baseline_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(8*1024*1024),b''):h.update(v)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
 p={name:OLD/name for name in ('analyze_saved.py','analyze_previous_target.py','report.json','previous_target_report.json')}
 response,prior=read(p['report.json']),read(p['previous_target_report.json'])
 assert sha(p['report.json'])=='eae2117e05178804036a047f5b0232a781072f45a7c273834d056f3b15028f45'
 assert response['source_sha256']==sha(p['analyze_saved.py']) and prior['source_sha256']==sha(p['analyze_previous_target.py'])
 pins={}
 for report in (response,prior):
  assert report['saved_analysis_complete'] is True and report['model_calls']==report['native_steps']==report['optimizer_updates']==0
  for path,digest in report['input_sha256'].items():
   assert path not in pins or pins[path]==digest;pins[path]=digest
 for path,digest in pins.items():assert sha(path)==digest,path
 for label,result in response['results'].items():
  assert len(result['cells'])==54
  identities={(r['dataset'],r['phase'],r['group']) for r in result['cells']};assert len(identities)==54
  for r in result['cells']:assert np.isclose(r['total'],r['odd']+r['even'],rtol=1e-12,atol=1e-16)
  for k,field in [('objective','total'),('odd','odd'),('even','even')]:assert result[k]==np.mean([r[field] for r in result['cells']])
 assert len(prior['cells'])==15 and sum(r['rows'] for r in prior['cells'])==9904
 assert prior['projected_previous_action_balanced_MSE']==np.mean([r['previous_action_projected_normalized_MSE'] for r in prior['cells']])
 ratio=prior['projected_previous_action_balanced_MSE']/prior['final_model_balanced_MSE']
 final=response['results']['final_GPU32'];zero=response['results']['zero_response'];initial=response['results']['initial_GPU32']
 fitpath=NEW/'direct_target_full_state_student_v1/fit/report.json';fit=read(fitpath)
 assert fit['metrics']['final_GPU32']['nominal_objective']==prior['final_model_balanced_MSE']
 for r in prior['cells']:
  trained=fit['metrics']['final_GPU32']['nominal_cells'][r['dataset']*3+r['phase']]
  assert r['final_model_normalized_MSE']==trained['normalized_MSE'] and r['final_model_RMSE_rad']==trained['preclip_RMSE_rad']
 mapping=NEW/'direct_target_full_state_student_v1/source_snapshot_v1/direct_data.py'
 feature=NEW/'one_step_physical_student_v1/source_snapshot_v1/student_linear_runtime.py'
 goals=NEW/'one_step_physical_student_v1/source_snapshot_v1/gear_sonic/utils/g1_true23_mpc_student.py'
 seed=NEW/'bfm_entry250_actual_oracle_v1/source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_bfm_seed.py'
 for path in [*p.values(),mapping,feature,goals,seed,Path(__file__)]:pins[str(path)]=sha(path)
 result=dict(source_review_pass=True,saved_summary_review_pass=True,input_sha256=pins,
  response_report_sha256=sha(p['report.json']),previous_target_report_sha256=sha(p['previous_target_report.json']),
  full_state_objectives=dict(initial=initial['objective'],final=final['objective'],zero_response=zero['objective']),
  final_vs_zero_ratio=final['objective']/zero['objective'],final_odd_error_fraction=final['odd']/final['objective'],
  prior_hold_vs_head_nominal_MSE_ratio=ratio,group_ratios=response['comparison'],
  findings=['No substantive algebra or row-ordering error found. Producer load_data concatenates3057 original centers, PICO, then walk002 in the tested15-cell order.',
   'Finite target changes use native target span only. Endpoint and center f32 predictions widen before subtraction. Each of54 cells gets equal weight; six-group averages each cover9 cells.',
   'Pair odd/even identity is exact algebra for symmetric signs: mean of both squared errors equals odd squared plus even squared. No radius division or derivative magnitude claim.',
   'Previous-target reconstruction matches original LinearFeatures and GoalFeatures52:75. The f32 normalized prediction/label square then f64 mean is comparable to the producer f32 reduction up to roundoff, not byte-identical reduction.',
   'analyze_previous_target imports hashing helpers from analyze_saved without binding that helper in its original input map. This review binds both actual sources; no producer report is changed.'],
  interpretation=['All6 aggregate groups remain worse than zero response, despite full objective improving. Nominal point accuracy and directional feedback accuracy remain separate requirements.',
   'A prior-only hold is21.95times worse than the head on balanced nominal MSE; this rejects an untrained previous-target skip/hold claim, not the possible conditional value of a causal prior.',
   'The nearly98percent odd error fraction points to antisymmetric response mismatch. It cannot identify insufficient width, optimizer interference, or omitted causal context by itself.',
   'Zero response is a conditional saved-data baseline, not an available closed-loop controller. No stability inference follows from its lower MSE.'],
  next_design_limits=['Do not extend the unchanged underfitting objective blindly or lower native limits/parity checks.',
   'A one-time initial gradient-norm coefficient only balances initial aggregate gradient norms; it does not prove persistent per-group optimization balance.',
   'Adding prior/history as conditioning is a testable hypothesis. Existing poor prior-hold error does not disprove it; exact-feature conflict absence does not establish approximate observability.',
   'Wait for the actual47-row semantics/fixed-map results before selecting one structural change; preserve all comparisons against the existing frozen65000 baseline.'],
  scope='Source review, actual input hashes, and saved report/cell arithmetic only; original full-array calculations were not repeated.',
  model_calls=0,native_steps=0,optimizer_updates=0,new_training_selected=False)
 with (BASE/'review.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(review_sha256=sha(BASE/'review.json'),prior_ratio=ratio,odd_fraction=result['final_odd_error_fraction'],final_over_zero=result['final_vs_zero_ratio'])))
if __name__=='__main__':main()
