"""Read-only completion verification; preserves unavailable child exit status."""
import hashlib
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


receipt = read(BASE/'process/launch_receipt.json')
post = read(BASE/'process/postrun_hashes.json')
for path, digest in receipt['input_hashes'].items():
    assert sha(path) == digest == post[path]
final = read(BASE/'collection/report.json')
assert final['complete'] is True
assert final['completed_rows'] == final['selected_rows'] == final['actor_inference_calls'] == final['backward_inference_calls'] == 6847
assert final['physics_steps'] == final['optimizer_calls'] == 0 and final['fitting_launched'] is False
assert not (BASE/'collection/failure.json').exists()
rows_receipt=read(BASE/'collection/rows_complete.json')
assert rows_receipt['complete'] is True and rows_receipt['completed_rows']==6847
assert rows_receipt['manifest_sha256']==final['manifest_sha256']
absence=read(BASE/'process_absence_verification.json')
assert absence['all_absent'] is True and not absence['running']
cases = {}
for clip, count, stop in (('pico',5980,6230),('walk002',867,1117)):
    report_path = BASE/'collection'/clip/'report.json'
    label_path = report_path.with_name('labels.npz')
    report = read(report_path)
    assert report == final['cases'][clip]
    assert report['complete'] is True and report['samples'] == count
    assert sha(label_path) == report['labels_sha256']
    assert sha(report_path) == rows_receipt['case_report_sha256'][clip]
    with np.load(label_path,allow_pickle=False) as data:
        assert data['complete'].item() is True
        np.testing.assert_array_equal(data['control'],np.arange(250,stop,dtype=np.int64))
        np.testing.assert_array_equal(data['source_frame'],np.arange(261,stop+11,dtype=np.int64))
        assert data['features'].shape==(count,1069) and data['state'].shape==(count,52)
        assert data['control_integration_before'].shape==(count,291)
        assert data['history'].shape==(count,300) and data['teacher_qpos'].shape==(count+1,30)
        assert np.isfinite(data['features']).all() and np.isfinite(data['residual_rad']).all()
        arrays = {k:dict(shape=list(data[k].shape),dtype=str(data[k].dtype)) for k in data.files}
    cases[clip] = dict(labels_path=str(label_path),labels_sha256=sha(label_path),report_sha256=sha(report_path),arrays=arrays)
compat = read(BASE/'collection/compatibility/report.json')
assert sha(BASE/'collection/compatibility/report.json') == final['bounded_compatibility_report_sha256']
assert compat['total_rows']==9904 and compat['new_rows']==6847
exit_record=read(BASE/'process/exit.json')
result=dict(kind='independent_saved_output_completion_verification_without_inference',
    producer_content_complete=True,expected_rows=6847,all136_launch_pins_unchanged=True,
    child_exit_status_recorded=exit_record['exit_code'],child_exit_status_was_unavailable=exit_record['exit_code'] is None,
    preserved_exit_receipt_sha256=sha(BASE/'process/exit.json'),
    all_wrapper_child_WSL_processes_absent=True,
    process_absence_verification_sha256=sha(BASE/'process_absence_verification.json'),
    rows_complete_receipt_sha256=sha(BASE/'collection/rows_complete.json'),
    final_report_sha256=sha(BASE/'collection/report.json'),cases=cases,
    compatibility_report_sha256=sha(BASE/'collection/compatibility/report.json'),
    exact_target_conflict_groups=compat['exact_target_conflict_groups'],
    exact_residual_conflict_groups=compat['exact_residual_conflict_groups'],
    inference_calls=0,physics_steps=0,fitting_launched=False,
    root_independent_label_audit_pending=True,source_sha256=sha(__file__))
(BASE/'completion_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('producer_content_complete','child_exit_status_was_unavailable','final_report_sha256',
    'compatibility_report_sha256','exact_target_conflict_groups','exact_residual_conflict_groups')}))
print(json.dumps({clip:{k:value[k] for k in ('labels_sha256','report_sha256')} for clip,value in cases.items()}))
