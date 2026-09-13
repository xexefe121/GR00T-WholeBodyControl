"""Prepare the completed-evidence reviewer; this does not open future outputs."""
from pathlib import Path
import ast
import difflib
BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
original=BASE/'direct_target_final_fit_review_v1/verify.py'
out=Path(__file__).resolve().parent
source=original.read_text()
changes=[
("BASE=NEW/'direct_target_student_v1'","BASE=NEW/'direct_target_continuation_v1'"),
("tracked={}","tracked={}\nimport argparse\nparser=argparse.ArgumentParser()\nfor key in ('fit_report_sha','checkpoint_sha','head_sha','root_audit','root_audit_sha','owner_sha'):parser.add_argument('--'+key.replace('_','-'),required=True)\nargs=parser.parse_args()"),
('c77506075081e608133eba7921e9ff6842464e069b31e01bafae12d9f0442a0e','e3dbc8c41433699326cd611ea97767974b6eb9ce5de2c7b1f678f83594054625'),
('ff918ef575c276bc4ed0ee262217163f5c65146faee1829d791d8ad793a6b226','1b9499a88b26b9feb5e1259d63284010e7521150c591a9815063cacc0251fa9c'),
('direct_target_fit_prelaunch_review_v1','direct_target_continuation_prelaunch_review_v1'),
('9b20a49687f1da2311975b5d7aff93d720bac55b2ed2c15724f4c8d05ca63f71','c1f0ab3a53503985bde58f7ab566a5ccb656af41d1d9e9c8c73f8f61320a120f'),
("'1d2f6f0e8fbee8e165b731c603f241885cb8f96ab385d9753feecd682011fd59'",'args.fit_report_sha'),
("'ad6d657affba0796e7313d85ace240cd31d46e188c577da049880a7a031aecba'",'args.checkpoint_sha'),
("'d35bf48cc1f755edef14dbe47a84f912ab2775c3a29a315f375d1c6d1918e088'",'args.head_sha'),
("audit_path=NEW/'direct_target_root_audit_v1/results_v1/report.json'","audit_path=Path(args.root_audit)"),
("'82a22c814def221dc8f11a14e53ba96004308635b1c3bd76a38cb256e6ae14b8'",'args.root_audit_sha'),
('direct_target_root_audit_source_review_v1','direct_target_continuation_root_audit_source_review_v1'),
('3f169d34dc6e245f7655a3fdb51b908fb623c3cad38f155a9ff7881a7eaefc6a','ccc24870b1c5bc422b439c9c32e9a95181f5800ab838d042e66eb938b7826581'),
('direct_target_root_audit_v1/audit_saved_fit.py','direct_target_continuation_root_audit_v1/audit_saved_fit.py'),
('40a47fd2e7554368c38a262fc9a2bdd0e75a3ceaa94ca817fbfb8646e191a55b','d4055bfea1fd7d7a0813954cc8f6135f43466ed021160e781a6ef5d6b8c829a5'),
("audit['checks']==16812","audit['checks']>=50000"),
("'export_parity_passed','fresh_initialization','all_frozen_inputs_unchanged'","'export_parity_passed','all_frozen_inputs_unchanged'"),
("assert report['ordinary_final_step']==report['additional_updates']==5000","assert report['ordinary_final_step']==55000 and report['additional_updates']==50000\nassert report['fresh_initialization'] is False and report['restored_start_step']==5000\nrestoration=read(fit/'restoration.json',report['restoration_sha256'])\nassert restoration['all_restoration_checks_passed'] is True and restoration['initial_GPU_predictions_exact'] is True and restoration['initial_metrics_exact'] is True"),
('70550000','705500000'),
("owner=read(BASE/'owner_completion_verification.json')","owner=read(BASE/'owner_completion_verification.json',args.owner_sha)"),
("==len(post['files'])==180","==len(post['files'])==202"),
("dataset_names=('centers','velocity_features','velocity_target','physical_manifest','pico','walk002')","dataset_names=('centers','velocity_features','velocity_target','physical_manifest','pico','walk002')"),
("fit_report=subject(fit/'report.json'),checkpoint=subject(checkpoint),head=subject(head),root_audit=subject(audit_path),","training_manifest=subject(BASE/'training_frozen_inputs.json'),restoration=subject(fit/'restoration.json'),\n    fit_report=subject(fit/'report.json'),checkpoint=subject(checkpoint),head=subject(head),root_audit=subject(audit_path),"),
("assert changes['full_velocity_objective']>0","# Preserve all objective tradeoffs; no monotonicity or best-checkpoint selection."),
("training_dataset_sha256=[subjects[name]['sha256'] for name in dataset_names]","training_dataset_sha256=[subjects['training_manifest']['sha256']]"),
('ordinary_final_step=5000,features=1000','ordinary_final_step=55000,additional_updates=50000,restored_start_step=5000,features=1000'),
('The root-selected ordinary final 5000 is retained; no checkpoint selection or additional fit.','The root-selected ordinary final55000 after exact5000 restoration is retained; no checkpoint selection or additional fit.'),
('Full velocity objective worsened from fresh initialization; all outcomes retained. No monotonic-loss requirement was selected.','All objective tradeoffs are retained. No monotonic-loss requirement was selected; this continues the exact ordinary5000 checkpoint.'),
]
for old,new in changes:
    assert old in source,old
    source=source.replace(old,new)
ast.parse(source)
with (out/'verify.py').open('x',encoding='utf-8') as f:f.write(source)
with (out/'source_derivation.patch').open('x',encoding='utf-8') as f:f.writelines(difflib.unified_diff(original.read_text().splitlines(True),source.splitlines(True),fromfile='direct5000/verify.py',tofile='continuation55000/verify.py'))
print('Prepared final55000 evidence reviewer; no future outputs opened.')
