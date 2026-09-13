from pathlib import Path
from datetime import datetime,timezone
NEW=Path(__file__).resolve().parent.parent;ART=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911')
stamp=datetime.now(timezone.utc).isoformat()
for name in ('CURRENT.md','SIM_RESULT.md'):
    with (Path(__file__).parent/(name.replace('.md','')+'_before_witness71000.md')).open('x',encoding='utf-8') as f:f.write((ART/name).read_text(encoding='utf-8-sig'))
p=ART/'CURRENT.md';s=p.read_text(encoding='utf-8-sig')
s=s.replace('Saved-fit audit v3 ACTIVE, one selected dispatch19:31:48UTC wrapper23048, reviewer soleowner.','Saved-fit audit v3 COMPLETE PASS66,447checks, raw/wrapper0/PIDs23048+28556absent/all38pins. Report427a91b32475e7f59212954d109865227f6b523c3e43ea9037c14462300a50ff,owner0899cdf7b9d56481ab71ea4d9c61f86d1104c4d5238d8df4d7e1c5e4e5f7d7a3. Evidence/exporttrue, no actualmodel/nativecalls. Do not repeat. Original selecteddispatch19:31:48UTC wrapper23048.')
s=s.replace('Root final release writer prepared, not executed until audit and owner pass.','Root finalrelease0ae9699fcb2b70ff33948223e9630a0f8692751baf2fdb27bbb191e7d76d5480 binds actual16roles plus completeaudit/fitowner.')
s=s.replace('No witness/canonical binding or actual controller selected until saved-fit audit.','ONE71000witness ACTIVE dispatched19:38:42UTC wrapper28648, expertsoleowner. Binding2a56cf0ab008f866625a336c8d6445fec2cebcfb4d55f26e8cf3995c0fa33f2c/launch75ce87dbc5ffcbf83ffdaff719be5194bf1c4fe2b76a7dfc5b9ebd2eeb07255c,5231pins; independentconcrete20268e368ad0dc9453a355f28288b85d8f495b3ee00e3f156161aa7dd3d00b46; clearance4c74b572f7fbbab82b6b8a33edae2356cb968f3ebf9ec4d80407ba968cbcb753. Exactly1WSLhead/0BFM/0native; do notduplicate. Canonical1569+conditional250 pendingwitnessowner/concretereview.')
s+='''
PendingBUSY clock concrete rootPASS NEW/independent_pending_clock_concrete_root_review_v1/review.json60bde676ea9e8c44d6c3abf769a9beb5bd852532d65201bff87a3754a87d3ce0,request2b538b03eb25de6d4c2ea3165561371af2415089957f4b9dd0bf9020a893118c,launchc7cc875c5f4c1d36e9efef5a56c8091d638e66d48393a0086bb2b30681c4dc3e;3768pins/3689originalexternal/12tests/exactargs. Execution_selectedfalse, noclearance. Matching savedhelper rootc2e17f4e4eb28c8cdb4aa95e3deb9fb2cf8beb20cedff4b0cf382caee53b44b0/10tests passed. Runclockonlyafter71000canonical+rootphysics completes andbeforewidthfit; Pico futureowner. Width27synthetic preparation underindependentPicosourcereview; noactualwidthfitselected.
'''
p.write_text(s,encoding='utf-8')
p=ART/'SIM_RESULT.md';s=p.read_text(encoding='utf-8-sig')
start=s.index('First independent saved-fit audit stopped');end=s.index('Recorded-command clock benchmark',start)
s=s[:start]+'''Independent saved-fit audit now passed66,447checks, with complete export and process verification. Earlier metadata-rule and CPU/GPU reduction-order checker failures remain preserved; trained model unchanged. Exactly one WSL activation check started19:38UTC. Full canonical walk003 simulation follows only after its completion and concrete check. No new stability result yet.

'''+s[end:]
p.write_text(s,encoding='utf-8')
with (ART/'SESSION.md').open('a',encoding='utf-8') as f:f.write('\n\n'+stamp+': savedfitv3PASS66447/report427a91b3/owner0899cdf7; finalrelease0ae9699f; ONEwitness71000ACTIVE19:38:42wrapper28648/expertowner. Clockconcrete60bde676+savedhelperc2e17f4passed butunselected.\n')
print('Current simulation status updated.')
