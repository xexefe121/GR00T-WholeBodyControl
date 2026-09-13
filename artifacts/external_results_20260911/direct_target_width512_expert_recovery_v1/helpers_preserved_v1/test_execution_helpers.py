"""No process/native/model execution. Literal metadata and owner map checks."""
import ast
from pathlib import Path
import pytest
from verify_recovery_completed import union,validate_counters
from prepare_execution_packet import wsl_path
BASE=Path(__file__).parent

def test_pin_union_rejects_alias_omission_and_preserves_values(tmp_path):
    a=tmp_path/'a';a.write_text('a')
    result=union({str(a):'hash'})
    assert len(result)==1 and list(result.values())==[(str(a),'hash')]
    with pytest.raises(ValueError):union({str(a):'hash'},{str(a.parent/'x'/'..'/'a'):'hash'})

def test_owner_requires_exact_membership_and_current_hashes():
    source=(BASE/'verify_recovery_completed.py').read_text()
    assert "actual.keys()==pins.keys()" in source
    assert "for p,digest in pins.values():assert sha(p)==digest,p" in source
    assert "counts['phase']=='task_ended'" in source
    assert "if not final:uncertainty.append" in source
    assert "absence['windows_expected_pids']==windows" in source
    assert "absence['linux_expected_pids']==linux_ids" in source
    assert 'returned_calls\']==value[\'attempted_calls' not in source

def test_launcher_hidden_required_clearance_and_known_exit():
    source=(BASE/'run_recovery_durable.ps1').read_text()
    for part in ('[Parameter(Mandatory=$true)][string]$ClearanceSha256','[IO.FileMode]::CreateNew',
                 '-WindowStyle Hidden -PassThru','[IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete',
                 '$handle=$child.Handle','$child.WaitForExit();$raw=$child.ExitCode',
                 '$clearance.launch_receipt_sha256','-eq $true','final_work_counters_present=$countersFinal','automatic_retry=$false'):
        assert part in source

def test_linux_identity_before_single_exec():
    text=(BASE/'run_recovery.sh').read_text()
    assert 'set -o noclobber' in text and text.count('\nexec ')==1
    assert text.index('linux_process.json')<text.index('\nexec ')
    assert 'source_snapshot_v1/run_width251_actual_oracle.py' in text

def test_request_builder_is_no_dispatch():
    source=(BASE/'prepare_execution_packet.py').read_text();tree=ast.parse(source)
    assert 'subprocess' not in source and 'Popen' not in source
    assert "actual_recovery_dispatched=False" in source
    assert "dispatch_authorized=False" in source
    assert "source_snapshot_v1" in source and "root_selected=True" in source
    assert tree

def test_literal_windows_paths_convert_before_drive_mapping():
    assert wsl_path(r'E:\folder\package\source_snapshot_v1')=='/mnt/e/folder/package/source_snapshot_v1'
    assert wsl_path('Z:/codex/tree/RUN_PASSING_WALK_WSL.sh')=='/mnt/z/codex/tree/RUN_PASSING_WALK_WSL.sh'
    assert '\\' not in wsl_path(r'E:\a\b')
    with pytest.raises(ValueError):wsl_path('relative/path')

def test_native_batch_uncertainty_is_not_high_level_seed_rejection():
    names=['native_private_step','native_BFM_step','native_initial_certificate_step','native_restoration_certificate_step',
           'native_preview_step','batch_fd_step','batch_line_step','BFM_propose']
    caps={k:dict(attempted_calls_max=10,attempted_units_max=100) for k in names}
    values={k:dict(attempted_calls=0,returned_calls=0,attempted_units=0,returned_units=0) for k in names}
    values['BFM_propose'].update(attempted_calls=1,attempted_units=1)
    counts=dict(limits=caps,counts=values)
    assert validate_counters(counts,caps)==[]
    values['batch_fd_step'].update(attempted_calls=1,attempted_units=82)
    assert 'internal completed lane-steps unknown' in validate_counters(counts,caps)[0]
    values['BFM_propose']['returned_calls']=False
    with pytest.raises(AssertionError):validate_counters(counts,caps)

def test_owner_rejects_missing_or_self_declared_limits():
    caps={'x':dict(attempted_calls_max=1,attempted_units_max=1)}
    with pytest.raises(AssertionError):validate_counters(dict(limits=caps,counts={}),caps)
    with pytest.raises(AssertionError):validate_counters(dict(limits={},counts={}),caps)
