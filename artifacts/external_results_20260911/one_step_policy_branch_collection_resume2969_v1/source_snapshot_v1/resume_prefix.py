"""Restore copied committed evidence only; no native, actor or head execution."""
import shutil
from pathlib import Path
import numpy as np
from collection_arrays import sha,atomic
from branch_inputs import read,exact
PREFIX_ROWS=2969
PREFIX_STEPS=29690
def restore_prefix(store,inputs,paths,destination):
    prior=read(paths['prefix_data']/'manifest.json');audit=read(paths['prefix_verification']);failure=read(paths['prefix_failure'])
    assert audit['passed'] and audit['verified_nominal_prefix']==PREFIX_ROWS and audit['recorded_native_steps']==PREFIX_STEPS
    assert audit['manifest_sha256']==sha(paths['prefix_data']/'manifest.json') and audit['failure_sha256']==sha(paths['prefix_failure'])
    assert failure['progress']['native_attempted']==failure['progress']['native_returned']==PREFIX_STEPS
    assert failure['progress']['nominal_verified']==PREFIX_ROWS and failure['progress']['active_row']==PREFIX_ROWS
    for value in failure['progress']['calls'].values():assert value==dict(attempted=0,returned=0)
    copied={}
    for key,spec in prior['arrays'].items():
        digest=sha(store.folder/spec['path']);assert digest==spec['sha256'];copied[key]=digest
    mask=np.arange(3054)<PREFIX_ROWS
    exact(store.arrays['nominal_verified'],mask,'copied verified prefix')
    exact(store.arrays['nominal_status'],mask.astype(np.int32),'copied native status')
    for key in ('valid_steps','attempted_steps','returned_steps'):exact(store.arrays['nominal_'+key],mask.astype(np.int32)*10,key)
    assert not store.arrays['policy_status'].any() and not store.arrays['label_valid'].any()
    for key in ('qpos','qvel','time','command_torque','actuator_force','start_integration','end_integration'):
        assert np.isnan(store.arrays['nominal_'+key][PREFIX_ROWS:]).all()
        assert np.isnan(store.arrays['policy_'+key]).all()
    for row,(dataset,control,index) in enumerate(inputs.rows):
        assert store.arrays['dataset'][row]==dataset and store.arrays['start_control'][row]==control and store.arrays['center_index'][row]==index
        assert store.arrays['successor_control'][row]==control+1 and store.arrays['successor_center_index'][row]==index+1
    for row,(dataset,control,index) in enumerate(inputs.rows[:PREFIX_ROWS]):
        expected=inputs.expected(inputs.traces[dataset],control)
        for key,value in expected.items():exact(store.arrays['nominal_'+key][row],value,f'restored sample {row} {key}')
        exact(store.arrays['nominal_start_integration'][row],inputs.integration(dataset,control),f'restored start291 {row}')
        exact(store.arrays['nominal_end_integration'][row],inputs.integration(dataset,control+1),f'restored end291 {row}')
    assert sha(paths['prefix_native_rows'])==audit['native_rows_sha256']
    shutil.copyfile(paths['prefix_native_rows'],destination/'native_rows.jsonl')
    assert sha(destination/'native_rows.jsonl')==audit['native_rows_sha256']
    report=dict(passed=True,copied_array_sha256=copied,prefix_rows=PREFIX_ROWS,prefix_native_steps=PREFIX_STEPS,
        source_manifest_sha256=sha(paths['prefix_data']/'manifest.json'),source_failure_sha256=sha(paths['prefix_failure']),
        native_ledger_prefix_sha256=audit['native_rows_sha256'],native_ledger_prefix_bytes=paths['prefix_native_rows'].stat().st_size,
        first_new_nominal_row=PREFIX_ROWS,remaining_nominal_rows=85,all_policy_rows_unexecuted=True,
        no_native_or_graph_calls_during_restore=True,original_attempt_unchanged=True)
    atomic(destination/'restored_prefix.json',report);return report
