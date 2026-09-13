"""Prepare one existing witness or full-motion invocation; never dispatches."""
import argparse,json,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'source_draft_v1'))
from evaluation_gate import read,sha
from freeze_final_package import write_new
from prepare_bound_launcher import run_text,durable_text,wsl_arguments
p=argparse.ArgumentParser();p.add_argument('mode',choices=['witness','evaluation']);args=p.parse_args();mode=args.mode
folder=BASE/(mode+'_process');assert not folder.exists()
for script,argv in [('freeze_final_package.py',['--mode',mode,'--reviews',str(BASE/'release_reviews.json')]),
                    ('prepare_bound_launcher.py',['--mode',mode])]:
    subprocess.run([sys.executable,'-B',str(BASE/script),*argv],cwd=BASE,check=True)
binding_path=BASE/(mode+'_binding.json');receipt_path=folder/'launch_receipt.json'
binding=read(binding_path);receipt=read(receipt_path)
assert receipt['binding_sha256']==sha(binding_path) and receipt['exact_wsl_arguments']==wsl_arguments(BASE,mode)
assert (folder/'run.ps1').read_text()==run_text(BASE,folder,mode,sha(binding_path))
assert (folder/'run_durable.ps1').read_text()==durable_text(folder)
for path,digest in receipt['input_hashes'].items():assert sha(path)==digest,path
assert receipt['requested_main_controls']==(1569 if mode=='evaluation' else 0)
assert receipt['conditional_hold_controls']==(250 if mode=='evaluation' else 0)
assert receipt['expected_separate_head_calls']==(1 if mode=='witness' else 0)
review_path=folder/'root_concrete_review.json'
write_new(review_path,dict(passed=True,binding_subject=dict(path=binding_path.as_posix(),sha256=sha(binding_path)),
    launch_receipt_subject=dict(path=receipt_path.as_posix(),sha256=sha(receipt_path)),
    all_pins_exact=True,checked_pins=len(receipt['input_hashes']),original_control_and_physics_code_unchanged=True,
    native_steps=0,model_calls=0,actual_dispatch=False,physical_qualification=False))
subprocess.run([sys.executable,'-B',str(BASE/'prepare_bound_launcher.py'),'--mode',mode,'--final-review',str(review_path)],cwd=BASE,check=True)
print(json.dumps({'mode':mode,'clearance_sha256':sha(folder/'launch_clearance.json')}))
