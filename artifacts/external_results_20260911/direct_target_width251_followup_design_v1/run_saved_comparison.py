"""One saved-only comparison of the selected first fresh expert target."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from compare_first_target import compare
HERE=Path(__file__).resolve().parent
NEW=HERE.parent
SEM=NEW/'direct_target_width512_saved_semantics_review_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def canonical(p):return Path(p).resolve().as_posix().casefold()
def load(p):
 with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def bound(p,pins):
 normalized={canonical(k):v for k,v in pins.items()}
 digest=sha(p)
 if normalized.get(canonical(p))!=digest:raise ValueError('Unbound saved subject: '+str(p))
 return digest
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--recovery-base',type=Path,required=True)
 parser.add_argument('--owner-file',required=True)
 parser.add_argument('--owner-sha',required=True)
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();base=args.recovery_base
 owner_path=base/args.owner_file
 assert sha(owner_path)==args.owner_sha
 owner=read(owner_path)
 assert owner['completion_accounting_passed'] and owner['processes_absent'] and owner['raw_exit_known']
 assert owner['all_postrun_pins_exact'] and owner['owner_task_model_calls']==owner['owner_native_steps']==0
 old_owner_path=SEM/'owner_completion.json'
 assert sha(old_owner_path)=='7d9fee34c3bae9f41d0a133e3cbc8593fd8b3a3ac7415dc627adfca183290ebd'
 old=read(old_owner_path)
 assert old['completion_accounting_passed'] and old['evidence_audit_passed'] and old['processes_absent']
 paths=dict(snapshot=base/'inputs/precontrol251.npz',first=base/'nominal/first_expert_target.npz',
  branch=base/'nominal/trace.npz',actual_rows=SEM/'results_v1/actual_rows.npz',
  fixed_maps=SEM/'results_v1/fixed_map_arrays.npz')
 bindings={}
 for key,p in paths.items():
  pins=old['output_sha256'] if key in ('actual_rows','fixed_maps') else owner['input_sha256'] if key=='snapshot' else owner['output_sha256']
  bindings[p.as_posix()]=bound(p,pins)
 assert bindings[paths['snapshot'].as_posix()]=='aa8cd94cdefc77b96624c75b4825a1133d805b8fd8be0717dc765e4c8f00c3c3'
 assert not args.output.exists(),'Preserve existing saved comparison'
 result=compare(**{key:load(p) for key,p in paths.items()})
 # No concurrent writer is permitted; recheck each exact input after the read.
 for p,digest in bindings.items():assert sha(p)==digest
 assert sha(owner_path)==args.owner_sha and sha(old_owner_path)=='7d9fee34c3bae9f41d0a133e3cbc8593fd8b3a3ac7415dc627adfca183290ebd'
 result.update(saved_comparison_pass=True,created_utc=datetime.now(timezone.utc).isoformat(),
  input_sha256=bindings,recovery_owner=dict(path=owner_path.as_posix(),sha256=args.owner_sha),
  semantics_owner=dict(path=old_owner_path.as_posix(),sha256=sha(old_owner_path)),
  source_sha256={p.name:sha(p) for p in (Path(__file__),HERE/'compare_first_target.py')},
  recovery_completed=owner['requested_recovery_completed'],recovery_qualification_inferred=False)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(passed=True,output=args.output.as_posix(),sha256=sha(args.output),
  fresh_minus_fixed_map=result['fresh_minus_fixed_map']['rmse_rad'],
  fresh_minus_student=result['fresh_minus_student']['rmse_rad'],
  fresh_target_was_issued=result['fresh_target_was_issued'],first_control_native_steps=result['first_control_native_steps'])))
if __name__=='__main__':main()
