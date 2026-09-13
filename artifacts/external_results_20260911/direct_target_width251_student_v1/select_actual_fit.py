"""Select the single already configured fit after checking its concrete inputs."""
from pathlib import Path
from datetime import datetime,timezone
import json
from prepare_execution import sha,read,write,build_request,BASE
request_path=BASE/'training_request.json';frozen_path=BASE/'training_frozen_inputs.json';launch_path=BASE/'launch_receipt.json'
assert sha(request_path)=='7d971b56d3746e582f5d9b014c45d98e07c4e71aa7ecf229ff1a86af281ebdb5'
assert sha(frozen_path)=='50b3958b202cc296505223ece86a1d9435915c037250bc234412565f1736e3c3'
assert sha(launch_path)=='fa899280f0ef2baeeefbe517c8e29b49e89c11267bb8c1201b6791a8cd6125a1'
request=read(request_path);frozen=read(frozen_path);launch=read(launch_path)
assert request==build_request(BASE/'selected_protocol.json',BASE/'root_review.json')
assert len(launch['input_sha256'])==launch['pin_count']==480
for path,digest in launch['input_sha256'].items():assert sha(path)==digest,path
assert frozen['source_sha256']==read(BASE/'source_preparation.json')['source_sha256']
assert not any((BASE/name).exists() for name in ('fit','fit_process_v1','training_clearance.json'))
clock_owner=BASE.parent/'independent_plant_timing_saved_actual_v1/owner_completion.json'
assert sha(clock_owner)=='bf90f9597feeae6c2932233e298599635a420b3e7fd57bab2662c8f661d2b92d'
assert read(clock_owner)['completion_accounting_passed'] is True
review_path=BASE/'concrete_root_review.json'
review=dict(prelaunch_review_pass=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
    training_request_sha256=sha(request_path),frozen_receipt_sha256=sha(frozen_path),
    launcher_sha256=launch['launcher_sha256'],launch_receipt_sha256=sha(launch_path),
    helper_preparation_sha256=sha(BASE/'execution_helper_preparation.json'),checked_pins=480,
    configuration_exact=True,ordinary_start_step=81000,ordinary_final_step=91000,optimizer_start_step=16000,
    optimizer_final_step=26000,updates=10000,recovery_coefficient=.2,all_inputs_exact=True,
    model_calls=0,native_steps=0,automatic_retry=False,behavioral_qualification=False,writer_sha256=sha(__file__))
write(review_path,review)
clear=dict(approved=True,request_sha256=sha(request_path),frozen_receipt_sha256=sha(frozen_path),
    launch_receipt_sha256=sha(launch_path),launcher_path=(BASE/'run_fit_durable_v1.ps1').as_posix(),launcher_sha256=launch['launcher_sha256'],
    review_path=review_path.as_posix(),review_sha256=sha(review_path),review_pass_field='prelaunch_review_pass',
    condition='causal',updates=10000,ordinary_start_step=81000,ordinary_final_step=91000,
    optimizer_start_step=16000,optimizer_final_step=26000,automatic_retry=False)
write(BASE/'training_clearance.json',clear)
print(json.dumps({'clearance_sha256':sha(BASE/'training_clearance.json'),'review_sha256':sha(review_path),'launch_receipt_sha256':sha(launch_path)}))
