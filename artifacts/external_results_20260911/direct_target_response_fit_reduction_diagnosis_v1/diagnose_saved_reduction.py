"""Saved 3,000x15 loss-cell arithmetic only; no task forward or optimizer."""
from pathlib import Path
from fractions import Fraction
import hashlib,json
import numpy as np
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
FIT=NEW/'direct_target_causal_response_balanced_student_v2';AUDIT=NEW/'direct_target_response_balanced_fit_independent_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    manifest=read(FIT/'fit/output_manifest.json')
    paths={n:FIT/'fit'/n for n in ('training_progress.npy','nominal_cell_losses.npy')}
    for n,p in paths.items():assert sha(p)==manifest['files'][n]
    progress=np.load(paths['training_progress.npy'],allow_pickle=False);cells=np.load(paths['nominal_cell_losses.npy'],allow_pickle=False)
    assert progress.shape==(3000,6) and cells.shape==(3000,15) and progress.dtype==cells.dtype==np.float64
    actual=progress[:,0];cells32=cells.astype(np.float32)
    assert np.array_equal(cells,cells32.astype(np.float64)) and np.array_equal(actual,actual.astype(np.float32).astype(np.float64))
    assert np.isfinite(cells).all() and np.all(cells>=0) and np.all(actual>0)
    mean32=cells32.mean(axis=1).astype(np.float64);mean64=cells.mean(axis=1)
    rational=np.array([float(sum((Fraction.from_float(v) for v in row),Fraction())/15) for row in cells],np.float64)
    eps=2.**-24;gamma16=(16*eps)/(1-16*eps);bound=gamma16*rational
    error32=np.abs(actual-mean32);relative32=error32/mean32
    error64=np.abs(actual-rational);relative64=error64/rational
    ulp=np.spacing(mean32.astype(np.float32)).astype(np.float64)
    failures=~np.isclose(actual,mean32,rtol=3e-7,atol=1e-14)
    assert np.allclose(actual,mean64,rtol=3e-7,atol=1e-14) and np.all(error64<=bound)
    rows=np.flatnonzero(failures)
    details=[dict(update=int(i+1),ordinary_step=int(68001+i),actual=float(actual[i]),numpy_float32_mean=float(mean32[i]),
        exact_rational_mean_rounded64=float(rational[i]),absolute_error_vs_numpy32=float(error32[i]),relative_error_vs_numpy32=float(relative32[i]),
        float32_ULPs_vs_numpy32=float(error32[i]/ulp[i]),relative_error_vs_exact_mean=float(relative64[i]),positive_sum_gamma16_bound=float(bound[i]),
        saved_float32_cells=cells[i].tolist()) for i in rows]
    tracked=[*paths.values(),FIT/'fit/output_manifest.json',FIT/'source_snapshot_v1/direct_objective.py',
        FIT/'source_snapshot_v1/train_response_balanced.py',AUDIT/'results_v1/report.json',AUDIT/'owner_completion.json',
        AUDIT/'source_prepared_v1/audit_saved_warm.py',AUDIT/'source_prepared_v1/audit_balanced_math.py',Path(__file__)]
    result=dict(diagnosis_passed=True,scope='saved nominal loss/cell reductions only; no audit rerun',rows=3000,cells_per_row=15,
        producer_expression='torch.stack(cell_losses).mean() in float32 on CUDA; saved entries are exact float32 values widened to float64',
        failing_auditor_expression='nominal.astype(np.float32).mean(axis=1).astype(np.float64) via NumPy CPU, compared with rtol3e-7',
        failure_rows=len(rows),first_failing_update=int(rows[0]+1) if len(rows) else None,
        max_abs_vs_numpy32=float(error32.max()),max_relative_vs_numpy32=float(relative32.max()),max_float32_ULPs_vs_numpy32=float((error32/ulp).max()),
        max_abs_vs_exact_mean=float(error64.max()),max_relative_vs_exact_mean=float(relative64.max()),
        exact_rational_mean_matches_float64_mean=bool(np.array_equal(rational,mean64)),
        original_GPU_vs_float64_mean_check_passes=True,all_actual_losses_float32_representable=True,all_saved_cells_float32_representable=True,
        all_rows_within_positive_float32_gamma16_bound=True,gamma16=gamma16,
        bound_reason='Conservative positive fifteen-term reduction: at most fourteen additions, rounded reciprocal, and multiplication. No cancellation; gamma16 times exact positive mean bounds rounding.',
        mismatches=details,recommendation='Preserve original established GPU-vs-float64 mean check; remove duplicate NumPy-float32 reduction-order expectation. Keep every cell, weight, total and model-export gate unchanged.',
        task_model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,saved_array_loads=2,
        input_sha256={p.as_posix():sha(p) for p in tracked})
    with (BASE/'report.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('failure_rows','first_failing_update','max_abs_vs_numpy32','max_relative_vs_numpy32','max_float32_ULPs_vs_numpy32','max_relative_vs_exact_mean')}))
    print('report_sha256',sha(BASE/'report.json'))
if __name__=='__main__':main()
