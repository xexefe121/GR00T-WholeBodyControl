"""Independent literal D3 qualification checks. Metadata only; no producer import."""
from pathlib import Path

RECOVERY_ROLES=('recovery_rows','collection_report','collection_request','collection_qualification','collection_source_review','consistency_report','warm_restore_review')

def path_key(path):
    value=str(path).replace('\\','/')
    if value.startswith('/mnt/') and len(value)>7 and value[6]=='/':value=value[5]+':'+value[6:]
    return value.casefold()

def exact_member(mapping,subject):
    normalized={}
    for path,digest in mapping.items():
        key=path_key(path)
        if key in normalized and normalized[key]!=digest:return False
        normalized[key]=digest
    return normalized.get(path_key(subject['path']))==subject['sha256']

def validate_recovery_bindings(request,read,bind,check,compare,sha,canonical,producer_sources):
    subjects=request['subjects']
    for role in RECOVERY_ROLES:bind(subjects[role]['path'],subjects[role]['sha256'])
    report=read(subjects['collection_report']['path']);selection=read(subjects['collection_request']['path'])
    for key,value in dict(passed=True,collection_completed=True,model_fitting_authorized=False,
            rows=1018,control_start=251,control_stop_exclusive=1269,fresh_student_state_queries=1).items():
        compare('collection.'+key,report[key],value)
    check('collection_request_identity',report['request_sha256']==subjects['collection_request']['sha256'])
    check('collection_rows_output_identity',report['outputs']['expert_rows.npz']==subjects['recovery_rows']['sha256'])
    check('collection_rows_output_path',canonical(subjects['recovery_rows']['path'])==canonical(Path(subjects['collection_report']['path']).parent/'expert_rows.npz'))
    check('collection_normalization_identity',report['outputs']['normalization.npz']==subjects['normalization']['sha256'])
    check('collection_selection_scope',selection['root_selected_collection'] is True and selection['model_fitting_authorized'] is False)
    for role,name in [('collection_qualification','qualification'),('collection_source_review','source_review')]:
        item=selection['subjects'][name]
        check('collection_literal:'+role,exact_member({item['path']:item['sha256']},subjects[role]) and exact_member(report['input_sha256'],subjects[role]))
    qualification=read(subjects['collection_qualification']['path'])
    for key,value in dict(root_authorized_extraction=True,model_fitting_authorized=False,control_start=251,control_stop_exclusive=1269,rows=1018).items():
        compare('collection_qualification.'+key,qualification[key],value)
    source=read(subjects['collection_source_review']['path'])
    check('collector_source_pass',source['source_review_pass'] is True)
    compare('collector_selected_source',selection['source_sha256'],source['source_sha256'])
    compare('collector_actual_source',report['source_sha256'],source['source_sha256'])
    consistency=read(subjects['consistency_report']['path'])
    for key,value in dict(passed=True,evidence_diagnosis_completed=True,old_rows=12958,new_rows=1018,rows=13976).items():
        compare('consistency.'+key,consistency[key],value)
    for role in ('recovery_rows','collection_report','collection_request','collection_qualification','collection_source_review','normalization'):
        check('consistency_literal:'+role,exact_member(consistency['input_sha256'],subjects[role]))
    warm=read(subjects['warm_restore_review']['path']);check('warm_restore_review_pass',warm['source_review_pass'] is True)
    # All producer source files are independently pinned by the enclosing frozen receipt.
    for name in ('warm512_restore.py','width512.py','restoration_support.py','balance_contract.py'):
        check('warm_source_pin:'+name,warm['source_sha256'][name]==producer_sources[name])
