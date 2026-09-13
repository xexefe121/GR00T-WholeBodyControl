"""Independent source review and synthetic probes only; no task data evaluation."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys
import numpy as np
BASE=Path(__file__).resolve().parent
NEW=BASE.parent
SUBJECT=NEW/'direct_target_full_state_data_root_review_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
if __name__=='__main__':
    preparation=json.loads((SUBJECT/'source_preparation.json').read_text())
    pins=dict(preparation['source_sha256'])
    for path,digest in pins.items():assert sha(path)==digest
    for path in [SUBJECT/'source_preparation.json',Path(__file__),NEW/'bfm_entry250_actual_oracle_v1/source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py']:
        pins[str(path)]=sha(path)
    with (BASE/'request.json').open('x') as f:f.write(json.dumps(dict(source_only=True,input_sha256=pins),indent=2)+'\n')
    with (BASE/'stdout.log').open('xb') as out,(BASE/'stderr.log').open('xb') as err:
        result=subprocess.run([sys.executable,'-B','-m','unittest','test_saved_math','-v'],cwd=str(SUBJECT),
                              env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=out,stderr=err)
    spec=importlib.util.spec_from_file_location('independent_review_saved_math',SUBJECT/'saved_math.py')
    math=importlib.util.module_from_spec(spec);spec.loader.exec_module(math)
    bad=np.zeros((1,23),np.float64);bad[0,0]=np.nan;index={}
    duplicates,conflicts=math.duplicate_conflicts(np.zeros((1,1000),np.float32),bad,np.ones(23),np.zeros(23),index,'synthetic')
    accepted_nonfinite=len(index)==1 and not duplicates and not conflicts
    assert accepted_nonfinite
    findings=[
        dict(id='producer-accounting',file='audit_saved_data.py',finding='No exact producer call_accounting or centers/signed/overlap/new report-counter checks.',required='Bind expected feature/map attempted/returned357669 and3057/354612/140622/213990 report totals; retain independent loop counts.'),
        dict(id='centers-output-membership',file='audit_saved_data.py',finding='generation/centers.npz is consumed without explicit report.output_sha256 membership; other actual arrays require it.',required='Require centers.npz output pin and bind before archive load.'),
        dict(id='nonfinite-label',file='saved_math.py',finding='Unique finite feature plus NaN target is silently indexed as a normalized label; synthetic probe reproduced.',required='Reject nonfinite targets and normalized float32 labels; verify23-value target schema and finite positive spans/defaults. Add synthetic regression.')]
    final={path:sha(path) for path in pins}
    review=dict(source_review_passed=False,findings=findings,existing_synthetic_tests_exit_code=result.returncode,
        existing_synthetic_tests_expected=7,synthetic_nonfinite_target_acceptance_reproduced=accepted_nonfinite,
        input_sha256=pins,all_inputs_unchanged=final==pins,source_preparation_sha256=sha(SUBJECT/'source_preparation.json'),
        test_stdout_sha256=sha(BASE/'stdout.log'),test_stderr_sha256=sha(BASE/'stderr.log'),
        correct_in_scope=['Fixed58 coordinate chart/group mapping and2e-12 change tolerance.','Literal native radii,58 axes x2 x3057=354612; old23 velocity axes140622, new35axes213990.',
            'Independent new feature source hash and original difference AST; no generator helper imported.','Raw feedback gain@tangent then feedback clamp+/-.1 and native clamp.',
            'Normalized absolute target label ((target-default64)/span32 promoted64).astype(float32); signed zeros canonicalized and raw rounding differences retained.',
            '9904nominal+354612fullstate+3054physical=367570, no averaging or row removal.'],
        task_array_loads=0,task_feature_evaluations=0,task_map_evaluations=0,model_calls=0,native_steps=0,optimizer_updates=0)
    with (BASE/'review.json').open('x') as f:f.write(json.dumps(review,indent=2)+'\n')
    print(json.dumps(dict(clear=False,tests_exit=result.returncode,all_inputs_unchanged=final==pins,review_sha256=sha(BASE/'review.json'))))
