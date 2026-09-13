"""Saved-byte review of corrected WSL transport; never starts a process."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from review_concrete import BASE, RUN, read, sha, subject, canonical, canonical_map

EXPECTED = {
    'execution_request.json':'c3e20cd213091bd9ec0db5a99871404ae609712de5be3e664b6bd844f62edea8',
    'frozen_inputs.json':'954ac6190406c575964db094fdd69d9fb387482a5ba2f19ed289b2194f698ce1',
    'launch_receipt_v2.json':'29484b217185eab4e5508dd2b524176fb56b21f3f601dcbf415bb62bb13e8fcc',
    'run_recovery_durable_v2.ps1':'261b0c3bd480e40d68241b60ba6fb2d7126721fbd525b41a9f87c2ba0e851a34',
    'verify_recovery_completed_v2.py':'2aa2e6508b06dc03e4d92d10451bb5ca955cbf496c79de24399225b0a913295c',
    'run_recovery_v2.sh':'157d169396b352aa6b6edb85a992d546fdbd1094e639d9ec8a1fe1fdd6128682',
    'argv_preflight_v2/report.json':'c2cb993824018b5c8b84f7ea2a2b6588a2b512cd9eebc8c2664e0490d747dd28',
    'execution_clearance.json':'29d674c341ae0264a44a2189a15740a88b3a5e218423173e4fa59743167d6260',
}

def main():
    for name,digest in EXPECTED.items(): assert sha(RUN/name)==digest,name
    launch=read(RUN/'launch_receipt_v2.json'); old=read(RUN/'launch_receipt.json')
    frozen=read(RUN/'frozen_inputs.json'); probe=read(RUN/'argv_preflight_v2/report.json')
    revision=read(RUN/'launcher_revision_v3.json')
    pins=canonical_map(launch['input_sha256']); assert len(pins)==103
    for path,digest in pins.values(): assert sha(path)==digest,path
    for path,digest in old['input_sha256'].items(): assert pins[canonical(path)][1]==digest,path
    for path,digest in frozen['input_sha256'].items(): assert pins[canonical(path)][1]==digest,path
    for name,digest in frozen['source_sha256'].items():
        assert pins[canonical(RUN/'source_snapshot_v1'/name)][1]==digest
    assert len(frozen['source_sha256'])==20
    assert launch['request_sha256']==EXPECTED['execution_request.json']
    assert launch['frozen_receipt_sha256']==EXPECTED['frozen_inputs.json']
    assert launch['launcher_sha256']==EXPECTED['run_recovery_durable_v2.ps1']
    assert launch['dispatch_authorized'] is False
    args=old['wsl_arguments'][:-1]+[old['wsl_arguments'][-1].replace('run_recovery.sh','run_recovery_v2.sh')]
    assert launch['wsl_arguments']==args
    expected_probe=args[:-1]+[args[-1].replace('run_recovery_v2.sh','argv_probe_v2.sh'),'FIXED_ARG_ONE','FIXED_ARG_TWO']
    assert probe['wsl_arguments']==expected_probe and probe['serialized_arguments']==' '.join(expected_probe)
    assert probe['passed'] is True and probe['powershell_version'].startswith('5.1.')
    assert probe['raw_exit_code']==0 and probe['handle_acquired'] and probe['windows_child_absent']
    assert all(x['rejected'] is True for x in probe['guard_tests']) and len(probe['guard_tests'])==3
    observed=read(RUN/'argv_preflight_v2/stdout.log'); assert observed==probe['linux_output']
    assert observed['cwd']=='/' and observed['argv']==['FIXED_ARG_ONE','FIXED_ARG_TWO']
    assert observed['probe_only'] is True
    for item in args[8:13]:
        key,value=item.split('=',1); assert observed[key]==value
    assert probe['task_python_calls']==probe['model_calls']==probe['native_steps']==0
    substitutions={'recovery_process_v1':'recovery_process_v2','launch_receipt.json':'launch_receipt_v2.json',
                   'execution_clearance.json':'execution_clearance_v2.json',
                   'run_recovery_durable.ps1':'run_recovery_durable_v2.ps1','owner_completion.json':'owner_completion_v2.json'}
    def renamed(text):
        for a,b in substitutions.items(): text=text.replace(a,b)
        return text
    prior_ps=renamed((RUN/'run_recovery_durable.ps1').read_text())
    new_ps=(RUN/'run_recovery_durable_v2.ps1').read_text()
    assignment=probe['actual_launcher_assignment']
    old_line=next(x for x in prior_ps.splitlines() if x.strip().startswith('$arguments='))
    new_line=next(x for x in new_ps.splitlines() if x.strip().startswith('$arguments='))
    assert new_line.strip()==assignment and prior_ps.replace(old_line,new_line)==new_ps
    assert renamed((RUN/'verify_recovery_completed.py').read_text())==(RUN/'verify_recovery_completed_v2.py').read_text()
    assert renamed((RUN/'run_recovery.sh').read_text())==(RUN/'run_recovery_v2.sh').read_text()
    for name in ('run_recovery_durable_v2.ps1','verify_recovery_completed_v2.py'):
        prior=(RUN/'v2_transport_before_gate_split'/name).read_text()
        assert prior.replace("'execution_clearance.json'","'execution_clearance_v2.json'")==(RUN/name).read_text()
    layers=launch['clearance_layers']; assert layers==revision['clearance_layers']
    for role in ('clearance','review'):
        item=layers['task_admission'][role]
        assert pins[canonical(item['path'])][1]==item['sha256']==sha(item['path'])
    assert layers['task_admission']['unchanged_request_and_frozen'] is True
    assert layers['task_admission']['previous_task_entry'] is False
    assert layers['root_must_select_corrected_transport'] is True
    assert canonical(layers['new_transport_clearance_path'])==canonical(RUN/'execution_clearance_v2.json')
    admission=(RUN/'source_snapshot_v1/recovery_admission.py').read_text()
    assert "'execution_clearance.json'" in admission and 'execution_clearance_v2.json' not in admission
    prior_exit=read(RUN/'recovery_process_v1/exit.json')
    assert prior_exit['raw_python_exit_code']==127 and prior_exit['exit_code']==1
    assert '-d: command not found' in (RUN/'recovery_process_v1/stderr.log').read_text()
    assert not (RUN/'recovery_process_v1/linux_process.json').exists()
    absent=['recovery_process_v2','execution_clearance_v2.json','owner_completion_v2.json','ATTEMPT_STARTED',
            'work_counters.json','driver_completion.json','nominal','post_lifecycle_hold_5s']
    for name in absent: assert not (RUN/name).exists(),name
    suites=ET.parse(BASE/'transport_v2_owner_tests.xml').getroot().findall('testsuite')
    assert sum(int(x.attrib['tests']) for x in suites)==8
    assert all(int(x.attrib.get(k,0))==0 for x in suites for k in ('failures','errors','skipped'))
    report=dict(passed=True,concrete_review_pass=True,transport_review_pass=True,
        request_sha256=EXPECTED['execution_request.json'],frozen_receipt_sha256=EXPECTED['frozen_inputs.json'],
        launch_receipt_sha256=EXPECTED['launch_receipt_v2.json'],launcher_sha256=EXPECTED['run_recovery_durable_v2.ps1'],
        launch_receipt_subject=subject(RUN/'launch_receipt_v2.json'),request_subject=subject(RUN/'execution_request.json'),
        frozen_subject=subject(RUN/'frozen_inputs.json'),revision_subject=subject(RUN/'launcher_revision_v3.json'),
        helper_sha256={n:EXPECTED[n] for n in ('run_recovery_durable_v2.ps1','run_recovery_v2.sh','verify_recovery_completed_v2.py')},
        input_sha256=launch['input_sha256'],launch_pin_count=103,source_count=20,
        prior_source_review=subject(BASE/'source_review.json'),prior_boundary_review=subject(BASE/'boundary_review.json'),
        prior_concrete_review=subject(BASE/'concrete_review.json'),clearance_layers=layers,
        actual_argv_proof=subject(RUN/'argv_preflight_v2/report.json'),wsl_arguments=args,
        tests=dict(passed=True,count=8,receipt=subject(BASE/'transport_v2_owner_tests.xml'),
                   source=subject(BASE/'test_owner_receipts_transport_v2.py')),
        preserved_first_attempt=dict(raw_wsl_exit=127,task_linux_identity_absent=True,task_entry_absent=True,
                                    original_owner_conservative_uncertainty_preserved=True),
        checks=dict(exact_original_request_frozen_and_twenty_sources=True,original_task_gate_unchanged_and_pinned=True,
                    new_transport_gate_separate=True,owner_only_path_derivative=True,real_ps51_argv_probe_pass=True,
                    actual_launcher_assignment_proved=True,all_current_pins_exact=True),
        processes_and_outputs_absent=absent,required_clearance_argument='-ClearanceSha256 actual execution_clearance_v2.json SHA256',
        task_model_calls=0,native_calls=0,replans=0,optimizer_calls=0,actual_recovery_dispatched=False,
        actual_recovery_execution_cleared=False,labels_admissible=False,
        limitations=['Root must select the corrected transport; expert retains sole dispatch ownership.',
                    'Original failed transport stays preserved; no task entry or numerical work was established there.',
                    'The shell-only argv proof verifies transport, not recovery physics, intent, timing or labels.'],
        review_source=subject(Path(__file__)))
    out=BASE/'transport_v2_review.json'
    with out.open('x',newline='\n') as f: json.dump(report,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps(dict(review=subject(out),passed=True,pins=103,tests=8)))

if __name__=='__main__': main()
