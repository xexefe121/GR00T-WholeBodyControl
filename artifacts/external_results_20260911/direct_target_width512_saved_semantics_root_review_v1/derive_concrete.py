from pathlib import Path
HERE=Path(__file__).resolve().parent;NEW=HERE.parent
s=(NEW/'direct_target_response_saved_semantics_root_review_v1/review_concrete.py').read_text(encoding='utf-8-sig')
mapping={
'direct_target_response_saved_semantics_review_v1':'direct_target_width512_saved_semantics_review_v1',
'one_pure_saved_causal71000_runtime_semantics_and_fixed_maps':'one_pure_saved_causal81000_width512_runtime_semantics_and_fixed_maps',
'3bf735731db9341ff63d01f8c7e4ad8eb1767f39d264ea34624d8babcb337671':'0d92c33169dd39c787037239fa27dd80fb90a17196163510f2f095f8dbe20e4d',
"stage['tests_run']==56":"stage['tests_run']==61",
'653fe9a1f06aa7138dad468891f823b71efa42fc9ed0ce11a83851b9c9c85fcc':'0caad5d8a7574bfd0a082be64b1f17c8cfd0be38360ef28bad767adfe6b3552a',
'd1ad17ef99dd0294b5ccc402c6073536b6e49ecafeea23a86dd1d004b7ea51eb':'d20ba43f42227258a26b293edef6134cc7772391cbeac4cdfa87f5127d31da20',
'acbcad444f1a8a6ba58c2aac22aeb82bfa6615afa7bcd0317abb13fbd8d66a95':'9d89c2456658a1df603a5b413786846a34a5f92fa140b82ebeda1d93db1e83af',
'78b2f9692b6de5d5d8b5310f9ac4e4803cc43a45532bb4ae0e79bea94f604801':'b9061a1c9dc6aee65d713f16909a3135f072d94a5bc20af750b5958cc493629d',
"physics['physics_steps']==2879":"physics['physics_steps']==3096",
"intent['recorded_controls']==288":"intent['recorded_controls']==310",
'Exact causal71000':'Exact causal81000 width512','completed root2879-step':'completed root3096-step'}
for a,b in mapping.items():assert a in s,a;s=s.replace(a,b)
with (HERE/'review_concrete.py').open('x',encoding='utf-8') as f:f.write(s)
