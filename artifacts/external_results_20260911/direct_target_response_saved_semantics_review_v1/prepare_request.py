"""Freeze actual completed subjects only. No evaluation or array math on import."""
import argparse
from saved_common import *

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--main-physics',type=Path,required=True);parser.add_argument('--main-physics-sha',required=True)
    parser.add_argument('--hold-physics',type=Path);parser.add_argument('--hold-physics-sha')
    parser.add_argument('--root-intent',type=Path,required=True);parser.add_argument('--root-intent-sha',required=True)
    parser.add_argument('--owner-sha',required=True);parser.add_argument('--source-review',type=Path,required=True);parser.add_argument('--source-review-sha',required=True)
    args=parser.parse_args();assert not (BASE/'request.json').exists() and not (BASE/'results_v1').exists()
    assert sha(args.source_review)==args.source_review_sha
    review=read(args.source_review);assert review['source_review_pass'] is True
    prep=read(BASE/'source_preparation.json')
    assert review['source_sha256']==prep['source_sha256']
    for name,digest in prep['source_sha256'].items():assert sha(BASE/name)==digest,name
    owner_path=RUN/'evaluation_completion_verification.json';assert sha(owner_path)==args.owner_sha
    owner=read(owner_path);assert owner['owner_completion_accounting_passed'] is True and owner['process_absence']['wrapper_absent'] is True and owner['process_absence']['child_absent'] is True
    binding=read(RUN/'evaluation_binding.json');assert binding['context_condition']=='causal' and binding['ordinary_final_step']==71000
    launch=RUN/'evaluation_process';clearance=read(launch/'launch_clearance.json')
    p=dict(binding=RUN/'evaluation_binding.json',owner=owner_path,launch=launch/'launch_receipt.json',posthash=launch/'postrun_hashes.json',
        process_start=launch/'start.json',process_child=launch/'child.json',process_exit=launch/'exit.json',process_raw=launch/'raw_exit.json',
        clearance=launch/'launch_clearance.json',canonical_review=local(clearance['review']['path']),frozen=RUN/'frozen_inputs_v1.json',
        witness=RUN/'head_witness/witness.npz',witness_report=RUN/'head_witness/report.json',witness_owner=RUN/'witness_completion_verification.json',
        contract=local(binding['contract']['path']),norm=local(binding['normalization']['path']),centers=local(binding['centers']['path']),labels=local(binding['query250_labels']['path']),
        motion=NEW.parent/'sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk003/reference.npz',
        baseline=NEW/'original_bfm_entry250_v1/entry250/trace.npz',
        feature_source=SOURCE/'direct_features.py',observation_source=SOURCE/'gear_sonic/utils/g1_true23_bfm_seed_observations.py',
        map_core=NEW/'bfm_entry250_actual_oracle_v1/source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py',
        map_original=NEW/'direct_target_fp64_fixed_map_v1/diagnose_fixed_map.py',
        source_preparation=BASE/'source_preparation.json',source_review=args.source_review)
    p['original29']=p['contract'].parent/'walk003/original29.npz'
    for name in SUBJECTS:p['subject_'+name]=local(binding[name]['path'])
    for name,entry in [('root_training_audit',binding['root_training_audit']),('fit_owner',binding['fit_owner_completion']),('final_release',binding['reviews']['release']),('source_review_runtime',binding['reviews']['source'])]:
        path=local(entry['path']);assert sha(path)==entry['sha256'];p['source_review' if name=='source_review_runtime' else name]=path
    p['audit_source_review']=args.source_review
    p['release_configuration']=RUN/'release_reviews.json'
    config=read(p['release_configuration']);helper_entry=config['launch_helper_review']
    p['helper_review']=local(helper_entry['path']);assert sha(p['helper_review'])==helper_entry['sha256']
    for name in read(p['helper_review'])['helper_sha256']:p['actual_helper_'+name]=RUN/name
    p['root_audit_owner']=p['root_training_audit'].parent.parent/'owner_completion.json'
    release=read(p['final_release']);assert sha(p['root_audit_owner'])==release['audit_owner_sha256']
    for path in SOURCE.rglob('*.py'):p['actual_source_'+path.relative_to(SOURCE).as_posix()]=path
    for name in PARITIES:
        if (RUN/(name+'.json')).exists():p[name]=RUN/(name+'.json')
    assert sha(args.root_intent)==args.root_intent_sha
    p['root_intent']=args.root_intent
    assert sha(args.main_physics)==args.main_physics_sha
    for name in ('main','hold'):
        folder=RUN/('nominal' if name=='main' else 'post_lifecycle_hold_5s')
        p[name+'_report']=folder/'report.json'
        if (folder/'trace.npz').exists():
            p[name+'_trace']=folder/'trace.npz';p[name+'_request']=folder/'request.json'
            physics=args.main_physics if name=='main' else args.hold_physics
            digest=args.main_physics_sha if name=='main' else args.hold_physics_sha
            assert physics is not None and sha(physics)==digest,'Every actual segment requires completed independent physics report'
            p[name+'_physics']=physics
            for failure in ('strict_failure_state.npz','rejected_precontrol.npz'):
                if (folder/failure).exists():
                    assert name+'_failure' not in p,'Multiple active failure types';p[name+'_failure']=folder/failure
    assert 'main_trace' in p
    for name in prep['source_sha256']:p['audit_source_'+name]=BASE/name
    for name in ('main','hold'):
        contains(owner['output_hashes'],p[name+'_report'],sha(p[name+'_report']))
        if name+'_trace' in p:contains(owner['output_hashes'],p[name+'_trace'],sha(p[name+'_trace']))
    request=dict(kind='one_pure_saved_causal71000_runtime_semantics_and_fixed_maps',
        paths={name:str(path) for name,path in p.items()},input_sha256={str(path):sha(path) for path in p.values()},
        source_review_sha256=args.source_review_sha,owner_sha256=args.owner_sha,
        model_calls=0,BFM_calls=0,native_steps=0,optimizer_updates=0,replans=0,
        context_condition='causal',public_features=1323,original_requested_controls=1569,conditional_continuous_hold_controls=250,
        fixed_map_scope='Only actually issued learned controls with existing unique same-clock query250 maps; missing coverage explicit, duplicate maps fail; no replanned expert truth.')
    write(BASE/'request.json',request);print(json.dumps(dict(request_sha256=sha(BASE/'request.json'),input_pins=len(request['input_sha256']))))

if __name__=='__main__':main()
