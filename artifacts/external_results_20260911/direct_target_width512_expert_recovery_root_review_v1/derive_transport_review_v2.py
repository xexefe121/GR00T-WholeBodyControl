"""Derive the second transport review without changing recovery computation."""
from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'review_concrete.py').read_text(encoding='utf-8')
replacements={
 "'launch_receipt.json'":"'launch_receipt_v2.json'",
 '7012ee94d666085f2c48256949a94f34392ccac6388e2ee928248c383ad536c0':'29484b217185eab4e5508dd2b524176fb56b21f3f601dcbf415bb62bb13e8fcc',
 'len(pins)==74':'len(pins)==103',
 "'run_recovery_durable.ps1'":"'run_recovery_durable_v2.ps1'",
 'f89670e1bd4749ae7b8f85a3d124235b5c2259a49d6e7e6f05d8f39945bcb836':'261b0c3bd480e40d68241b60ba6fb2d7126721fbd525b41a9f87c2ba0e851a34',
 "'verify_recovery_completed.py'":"'verify_recovery_completed_v2.py'",
 'f06fb5e730cdb3416c058719f91d27ad17ccc07784538371cd6f72c500e263b8':'2aa2e6508b06dc03e4d92d10451bb5ca955cbf496c79de24399225b0a913295c',
 '/run_recovery.sh':'/run_recovery_v2.sh',
 "'run_recovery.sh'":"'run_recovery_v2.sh'",
 "assert read(BASE/'launcher_parse.json')['passed'] is True":"assert b'\\r' not in (BASE/'run_recovery_v2.sh').read_bytes()",
 "'execution_clearance.json','ATTEMPT_STARTED'":"'execution_clearance_v2.json','ATTEMPT_STARTED'",
 "'recovery_process_v1','owner_completion.json'":"'recovery_process_v2','owner_completion_v2.json'",
 'checked_launch_pins=74':'checked_launch_pins=103',
 "OUT/'concrete_review.json'":"OUT/'concrete_review_v2.json'",
 'pins=74':'pins=103',
}
for old,new in replacements.items():
 assert old in s,old
 s=s.replace(old,new)
extra='''
# Preserve the failed transport and its task admission as immutable upstream
# evidence. A different, explicit root selection authorizes only the v2 transport.
assert sha(BASE/'execution_clearance.json')=='29d674c341ae0264a44a2189a15740a88b3a5e218423173e4fa59743167d6260'
assert sha(OUT/'concrete_review.json')=='7ada54d355e90aa7299bf0339582ece7cbb6feee617c57eda6fb74593f1b7489'
assert sha(BASE/'owner_completion.json')=='f39eefa242db95fe336d3712942d40974e6af7a9b8290eb17e83fc1cd00096e9'
assert read(BASE/'recovery_process_v1/exit.json')['raw_python_exit_code']==127
assert '-d: command not found' in (BASE/'recovery_process_v1/stderr.log').read_text(encoding='utf-8')
assert not (BASE/'recovery_process_v1/linux_process.json').exists()
assert sha(BASE/'argv_preflight_v2/report.json')=='c2cb993824018b5c8b84f7ea2a2b6588a2b512cd9eebc8c2664e0490d747dd28'
probe=read(BASE/'argv_preflight_v2/report.json')
assert probe['passed'] and probe['raw_exit_code']==0 and probe['handle_acquired'] and probe['windows_child_absent']
assert probe['powershell_version'].startswith('5.1.')
assert probe['wsl_arguments'][:-3]==l['wsl_arguments'][:-1]
assert probe['task_python_calls']==probe['model_calls']==probe['native_steps']==0
assert all(x['rejected'] for x in probe['guard_tests']) and len(probe['guard_tests'])==3
assert probe['linux_output']['cwd']=='/' and probe['linux_output']['argv']==['FIXED_ARG_ONE','FIXED_ARG_TWO']
launcher=(BASE/'run_recovery_durable_v2.ps1').read_text(encoding='utf-8-sig')
assert probe['actual_launcher_assignment'] in launcher
assert "'execution_clearance_v2.json'" in launcher and "'execution_clearance.json'" not in launcher
assert independent['transport_review_pass'] and independent['tests']['count']==8 and independent['tests']['passed']
assert independent['input_sha256']==l['input_sha256']
for key in ('revision_subject','actual_argv_proof'):
 sub=independent[key];assert sha(sub['path'])==sub['sha256']
for key in ('receipt','source'):
 sub=independent['tests'][key];assert sha(sub['path'])==sub['sha256']
'''
anchor='result=dict(passed=True,concrete_review_pass=True'
assert anchor in s;s=s.replace(anchor,extra+'\n'+anchor)
s=s.replace('actual_recovery_dispatched=False,review_task_model_calls=0,', "actual_recovery_dispatched=False,transport_revision=2,prior_task_admission_preserved=True,prior_task_entry=False,corrected_transport_selected=True,review_task_model_calls=0,")
with (p/'review_concrete_v2.py').open('x',encoding='utf-8') as f:f.write(s)
print(p/'review_concrete_v2.py')
