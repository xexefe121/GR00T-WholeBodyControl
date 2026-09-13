"""Prepare saved-only consistency request after completed qualified collection.

Reads JSON and hashes only; actual array processing remains a separate command.
"""
import argparse,hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
SOURCE=BASE/'source_prepared_v1';sys.path.insert(0,str(SOURCE))
from admission import report_gate,key,PHYSICAL_KEYS,ROLES
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def item(path):return dict(path=Path(path).resolve().as_posix(),sha256=sha(path))
def save(path,value):
 with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--collection-report-sha256',required=True);a=ap.parse_args()
 fit=NEW/'direct_target_causal_width512_student_v1'
 col=NEW/'direct_target_width251_collection_v1/actual_v1';cr=col/'results_v1/report.json'
 assert sha(cr)==a.collection_report_sha256
 root=BASE/'root_review.json';assert sha(root)=='1100d985008254ed4fb92363c0c0b0cb930ebf64ed34937d866fbca98d4404fa'
 paths=dict(old_fit_report=fit/'fit/report.json',old_fit_owner=fit/'owner_completion_verification.json',
  old_fit_audit=NEW/'direct_target_width512_fit_independent_v1/results_v1/report.json',
  old_training_request=fit/'training_request.json',old_training_manifest=fit/'training_frozen_inputs.json',
  old_shared_manifest=fit/'fit/shared/output_manifest.json',old_context_alignment=fit/'fit/shared/context_alignment.json',
  old_nominal_context=fit/'fit/shared/nominal_context.npy',old_physical_context=fit/'fit/shared/physical_context.npy',
  normalization=fit/'fit/shared/normalization.npz',export_source=fit/'source_prepared_v1/width512_promoted.py',
  new_rows=col/'results_v1/expert_rows.npz',collection_report=cr,collection_request=col/'request.json',
  collection_qualification=col/'qualification.json',
  collection_source_review=NEW/'direct_target_width251_collection_root_review_v1/review.json',source_review=root)
 old_request=read(paths['old_training_request'])
 for role,name in [('old_centers','centers'),('old_pico','pico'),('old_walk002','walk002'),('physical_manifest','physical_manifest'),('contract','contract')]:
  paths[role]=Path(old_request['paths'][name])
 assert set(paths)==set(ROLES)
 subjects={role:item(path) for role,path in paths.items()}
 manifest=read(paths['physical_manifest'])
 physical={role:item(paths['physical_manifest'].parent/manifest['arrays'][role]['path']) for role in PHYSICAL_KEYS}
 nonreports={'old_centers','old_pico','old_walk002','old_nominal_context','old_physical_context','normalization','new_rows','export_source','contract'}
 reports={role:read(path) for role,path in paths.items() if role not in nonreports}
 report_gate(subjects,reports,physical)
 source_map={p.name:sha(p) for p in SOURCE.glob('*.py')};assert source_map==reports['source_review']['source_sha256']
 packet=BASE/'actual_v1';assert not packet.exists(),'Preserve existing actual diagnosis'
 packet.mkdir()
 request=dict(root_selected_saved_diagnosis=True,subjects=subjects,physical_arrays=physical,source_sha256=source_map,
  output=(packet/'results_v1').as_posix(),model_calls=0,native_steps=0,optimizer_updates=0,
  scope=dict(old_nominal=9904,old_physical=3054,new_rows=1018,alias_modes=6,proximity_queries=72,candidate_block=256),
  preparation_source_sha256=sha(__file__),actual_numeric_arrays_read=False)
 save(packet/'request.json',request)
 print(json.dumps(dict(prepared=True,request=item(packet/'request.json'),subjects=len(subjects),physical_arrays=len(physical),
  actual_diagnosis_executed=False,root_concrete_review_pending=True)))
if __name__=='__main__':main()
