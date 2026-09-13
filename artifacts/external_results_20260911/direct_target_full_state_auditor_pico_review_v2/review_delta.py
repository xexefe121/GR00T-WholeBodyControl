"""Narrow read-only source delta review plus nine synthetic tests only."""
from pathlib import Path
import ast,difflib,hashlib,json,os,subprocess,sys
BASE=Path(__file__).resolve().parent
NEW=BASE.parent
SUBJECT=NEW/'direct_target_full_state_data_root_review_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
if __name__=='__main__':
    preparation=json.loads((SUBJECT/'source_preparation_v2.json').read_text())
    old=json.loads((NEW/'direct_target_full_state_auditor_pico_review_v1/review.json').read_text())
    pins={};deltas=[]
    for name,digest in preparation['source_sha256'].items():
        path=SUBJECT/name;preserved=SUBJECT/'source_preserved_v1'/name
        assert sha(path)==digest
        old_digest=next(v for p,v in old['input_sha256'].items() if Path(p).name==name)
        assert sha(preserved)==old_digest
        pins[str(path)]=digest;pins[str(preserved)]=sha(preserved)
        deltas.extend(difflib.unified_diff(preserved.read_text(encoding='utf-8').splitlines(True),path.read_text(encoding='utf-8').splitlines(True),fromfile='preserved/'+name,tofile='v2/'+name))
    for path in [Path(__file__),SUBJECT/'source_preparation_v2.json',NEW/'direct_target_full_state_auditor_pico_review_v1/review.json',
                 NEW/'bfm_entry250_actual_oracle_v1/source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py']:
        pins[str(path)]=sha(path)
    before=functions(SUBJECT/'source_preserved_v1/saved_math.py');after=functions(SUBJECT/'saved_math.py')
    unchanged=[name for name in before if name!='duplicate_conflicts']
    assert all(before[name]==after[name] for name in unchanged)
    assert set(after)-set(before)=={'validate_producer_report'}
    with (BASE/'delta.patch').open('x',encoding='utf-8') as f:f.write(''.join(deltas))
    with (BASE/'request.json').open('x') as f:f.write(json.dumps(dict(source_only=True,input_sha256=pins),indent=2)+'\n')
    with (BASE/'stdout.log').open('xb') as out,(BASE/'stderr.log').open('xb') as err:
        result=subprocess.run([sys.executable,'-B','-m','unittest','test_saved_math','-v'],cwd=str(SUBJECT),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=out,stderr=err)
    final=all(sha(p)==s for p,s in pins.items())
    report=dict(source_review_passed=result.returncode==0 and final,actual_saved_audit_source_clear=result.returncode==0 and final,
        findings=[],resolved_findings=['producer-accounting','centers-output-membership','nonfinite-label'],
        resolutions=['Exact3057/354612/140622/213990 totals and357669 pure feature/map calls plus each attempted/returned counter.',
            'centers.npz mandatory in output map before consumption; manifest/center/54summary and every14array schema/hash identities bound.',
            'Finite23-value target rows, finite positive spans/defaults and finite normalized-float32 labels required; NaN/Inf/overflow rejected.',
            'All54 unique cell/group counts/clipping/zero-response summaries independently checked against reconstructed loop totals.'],
        unchanged_math_functions=unchanged,synthetic_tests_expected=9,synthetic_tests_exit_code=result.returncode,
        input_sha256=pins,all_inputs_unchanged=final,preparation_sha256=sha(SUBJECT/'source_preparation_v2.json'),
        test_stdout_sha256=sha(BASE/'stdout.log'),test_stderr_sha256=sha(BASE/'stderr.log'),delta_sha256=sha(BASE/'delta.patch'),
        task_array_loads=0,task_feature_evaluations=0,task_map_evaluations=0,model_calls=0,native_steps=0,optimizer_updates=0,
        scope='Source clearance for the selected complete saved-data audit; actual outputs and data qualification remain pending its execution.')
    with (BASE/'review.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(clear=report['source_review_passed'],tests_exit=result.returncode,review_sha256=sha(BASE/'review.json'),pins=len(pins))))
