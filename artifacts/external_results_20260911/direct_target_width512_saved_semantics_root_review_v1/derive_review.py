from pathlib import Path
HERE=Path(__file__).resolve().parent;NEW=HERE.parent
s=(NEW/'direct_target_response_saved_semantics_root_review_v1/review_source.py').read_text(encoding='utf-8-sig')
mapping={
"BASE=NEW/'direct_target_response_saved_semantics_review_v1'":"BASE=NEW/'direct_target_width512_saved_semantics_review_v1'",
"OLD=NEW/'direct_target_context_saved_semantics_review_v1'":"OLD=NEW/'direct_target_response_saved_semantics_review_v1'",
"RUN=NEW/'direct_target_causal_response_evaluation_v1'":"RUN=NEW/'direct_target_causal_width512_evaluation_v1'",
'f92270c7e6a61eeec05c496860ce0f40ddf36b79ee9a4cb53644c3df2e72e148':'cff165de5b610c35ef7e973db7702944045fd615782bfaa6537aab14306eaaf3',
"prep['tests_run']==56":"prep['tests_run']==61","len(prep['source_sha256'])==15":"len(prep['source_sha256'])==16",
'direct_target_causal_response_evaluation_independent_review_v1':'direct_target_width512_evaluation_independent_review_v2',
'e40aca20c419219a8e6d1e4630ef5e4442ac7c1f492d85b009bc94b1460a0121':'db2a997d44634cb59d7d4bd92408fcf361323d143c6109024b457e5bd4a458de',
'29d74e990eca9aa861459ffb337bb140dfdafd338db1838e42f995d8494ff0d4':'883a0da493f3f0f5463f7d9a14c581adb840d5358e946b5db2b1b8962298d340',
'result.testsRun==56':'result.testsRun==61','actual_runtime_sources_exact=37':'actual_runtime_sources_exact=38','synthetic_tests=56':'synthetic_tests=61','tests=56':'tests=61',
'Sixteen actual71000 warm release roles; source68000, optimizer3000 to6000, unchanged coefficient/context; distinct producer and energy rule labels.':'Sixteen actual81000 width512 release roles; source71000, optimizer6000 to16000,10000 updates, seed20260912, split training/dense FP64 export, unchanged coefficient/context and distinct producer/energy rule labels.'}
for a,b in mapping.items():assert a in s,a;s=s.replace(a,b)
needle="assert functions(BASE/'saved_common.py')==functions(OLD/'saved_common.py')\n"
assert needle in s
s=s.replace(needle,needle+"assert functions(BASE/'prepare_audit_stage.py')==functions(OLD/'prepare_audit_stage.py')\n")
with (HERE/'review_source.py').open('x',encoding='utf-8') as f:f.write(s)
