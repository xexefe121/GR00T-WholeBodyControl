"""Future explicit width512 recovery91000 selection/review configuration, no placeholder subjects."""
import argparse,json
from pathlib import Path
from freeze_final_package import BASE,SOURCE,item,role,read,sha,field,bound_file,has_hash,actual_paths,base_binding,write_new,require_helper_review,require_condition
from context_release import SUBJECTS,validate_release


def configuration(*,root_selected,condition,release,root_audit,fit_owner,source_review,helper_review):
    if root_selected is not True:raise ValueError('parent has not selected this actual witness/canonical pipeline')
    condition=require_condition(condition)
    paths=actual_paths(condition);final={name:item(path) for name,path in paths.items()}
    config={'root_selected':True,'context_condition':condition,'selected_witness_calls':1,'selected_main_controls':1569,'conditional_hold_controls':250,
        'root_training_audit':role(root_audit),'fit_owner_completion':role(fit_owner),
        'reviews':{'release':role(release),'source':role(source_review)},'launch_helper_review':require_helper_review(helper_review),
        'subject_sha256':{name:final[name]['sha256'] for name in SUBJECTS}}
    inventory=read(BASE/'runtime_inventory.json');binding=base_binding(config,final,inventory)
    validate_release(binding,paths,SOURCE,'witness',read=read,sha=sha,bound_file=bound_file,field=field,has_hash=has_hash)
    return config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root-selected',action='store_true')
    parser.add_argument('--condition',choices=('causal',),required=True)
    for name,default in [('release','release_review_pass'),('root-audit','evidence_audit_passed'),
                         ('fit-owner','owner_verification_passed'),('source-review','passed'),('helper-review','passed')]:
        parser.add_argument('--'+name,type=Path,required=True);parser.add_argument('--'+name+'-pass-field',default=default)
    args=parser.parse_args()
    def spec(name):return {'path':str(getattr(args,name)),'pass_field':getattr(args,name+'_pass_field')}
    config=configuration(root_selected=args.root_selected,condition=args.condition,release=spec('release'),root_audit=spec('root_audit'),
        fit_owner=spec('fit_owner'),source_review=spec('source_review'),helper_review=spec('helper_review'))
    path=BASE/'release_reviews.json';write_new(path,config)
    print(json.dumps({'configuration_sha256':sha(path),'actual_subjects':len(config['subject_sha256']),
                      'head_calls':0,'native_steps':0,'launched':False}))


if __name__=='__main__':main()
