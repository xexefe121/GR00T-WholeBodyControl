"""One saved checkpoint metadata read; no model construction or numerical execution."""
from pathlib import Path
import ast,hashlib,json
import torch
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
FIT=NEW/'direct_target_causal_response_balanced_student_v2'
AUDIT=NEW/'direct_target_response_balanced_fit_independent_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    torch.set_num_threads(1)
    checkpoint_path=FIT/'fit/student_head.pt';report_path=FIT/'fit/report.json'
    assert sha(checkpoint_path)=='395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d'
    assert sha(report_path)=='f918f1dfe675aae01e25307d302ce6c54f599fb7f0f5d1fa3fd943a179b5aded'
    request=read(FIT/'training_request.json');energy_path=Path(request['subjects']['energy_source']['path'])
    energy=read(energy_path);report=read(report_path)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    contract_path=FIT/'source_snapshot_v1/balance_contract.py';tree=ast.parse(contract_path.read_text())
    rule=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='WEIGHT_RULE' for t in n.targets))
    actuals={'checkpoint':checkpoint['response_weight_rule'],'report':report['response_weight_rule']}
    actuals.update({label:value['response_weight_rule'] for label,value in report['metrics'].items()})
    assert rule=='mean_six_teacher_group_energies_over_group_energy' and all(v==rule for v in actuals.values())
    assert energy['rule']=='Emean/Eg'
    assert list(checkpoint['response_group_weights'])==energy['group_weights']==report['response_group_weights']
    failed=read(AUDIT/'results_v1/report.json');owner=read(AUDIT/'owner_completion.json')
    assert failed['evidence_audit_passed'] is False and failed['error']=="AssertionError('checkpoint_weight_rule')"
    assert owner['completion_accounting_passed'] is True and owner['processes_absent'] is True
    paths=[checkpoint_path,report_path,contract_path,energy_path,FIT/'training_request.json',
        AUDIT/'source_prepared_v1/audit_saved_warm.py',AUDIT/'source_prepared_v1/audit_balanced_math.py',
        AUDIT/'results_v1/report.json',AUDIT/'owner_completion.json',AUDIT/'process_v1/exit.json',Path(__file__)]
    result=dict(diagnosis_passed=True,cause='auditor_expected_energy_provenance_label_for_producer_metadata',
        producer_contract_rule=rule,actual_producer_fields=actuals,energy_provenance_rule=energy['rule'],
        all_six_actual_weights_match_energy_source=True,failed_audit_error=failed['error'],failed_audit_checks=failed['checks'],
        recommended_source_only_delta='Change the checkpoint/report expected metadata and reconstructed metric response_weight_rule to the literal producer contract. Retain energy.rule Emean/Eg and all weight arithmetic.',
        old_audit_preserved=True,actual_audit_retried=False,task_model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,
        saved_checkpoint_metadata_loads=1,input_sha256={p.as_posix():sha(p) for p in paths})
    with (BASE/'report.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(diagnosis_sha256=sha(BASE/'report.json'),failed_checks=failed['checks'],actual_rule=rule)))
if __name__=='__main__':main()
