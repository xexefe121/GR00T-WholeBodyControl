from pathlib import Path
OUT=Path(__file__).resolve().parent
NEW=OUT.parent
source=NEW/'independent_plant_clock_saved_actual_root_review_v1/review_concrete.py'
text=source.read_text(encoding='utf-8-sig')
changes={
"BASE=NEW/'independent_plant_clock_saved_actual_v1'":"BASE=NEW/'independent_plant_pending_publication_saved_actual_v1'",
'a4f6ddd9a8468c667e3de2b533d1f8098c7b93548c1c89de82909e839f30713f':'fba2b2841eab6a9f7697c472760e789224afd40ca71ca7c9c8692fd1b3251fdc',
'2579e1c45cc7686934fe344e8682ba8485b8aca4ad41cf1a098766fbd3c0bdf0':'63c8b002d6cd56cfc918cbc44f26b665cd0878240e4544c17ee30fac28cb685c',
'278c509418ef1c7a1be655693a840217dd8c1894a6a6c33f0757a712ad8b9ac7':'81c4090dd6035fc4b39064fd98e7412af87e2d0355a4a3abd260d19ffb4faae0',
"source=NEW/'independent_plant_clock_saved_root_review_v1/source_audit_v3'":"source=NEW/'independent_plant_pending_publication_saved_audit_v1/source_draft_v2'",
"assert len(r['source_sha256'])==14":"assert len(r['source_sha256'])==17",
"len(pins)==3802 and len(r['input_sha256'])==3792":"len(pins)==3827 and len(r['input_sha256'])==3817",
'3bf9f7ee97c7feb74434f1f928a33bb4827e62420a8f68f12f49ce5b73dd816e':'a51d7f88bff439a23c696ad48fa6cff5c8ba035cae58820cd1ac2c7f4ccf8b62',
'b63896a22b00a032f9d1ac5e9fcb420d796778607374d11e4a1ad7fec3ec52fc':'df8d562e18508b24372b81a7ecab6e8e0d53e195562078dc23ff6a45d54a89f6',
'0c22640dfb617a20c31aff7356434a6253096b0fc4cbed38aae64faeb14e75eb':'2b538b03eb25de6d4c2ea3165561371af2415089957f4b9dd0bf9020a893118c',
'f3eb4aca76e45b56761a95017090359a4de21d6edd1f3d7cacc71754158f1713':'c7cc875c5f4c1d36e9efef5a56c8091d638e66d48393a0086bb2b30681c4dc3e',
'unchanged reviewed v3 saved-only auditor, exact corrected clock run and final six stages':'Reviewed retry-aware v2 saved-only auditor, exact pending-BUSY clock run and final six stages',
"prep=read(BASE/'helper_preparation.json')":"""helper_review=NEW/'independent_pending_saved_helper_root_review_v1/review.json'
assert sha(helper_review)=='c2e17f4e4eb28c8cdb4aa95e3deb9fb2cf8beb20cedff4b0cf382caee53b44b0'
assert read(helper_review)['passed'] is True
assert sha(BASE/'helper_preparation.json')=='20407fd5dc10b837e8c8e87b94a69e337b5e066ed52398ac392147693097af18'
prep=read(BASE/'helper_preparation.json')""",
}
for a,b in changes.items():
    assert text.count(a)==1,(a,text.count(a))
    text=text.replace(a,b)
with (OUT/'review_concrete.py').open('x',encoding='utf-8') as f:f.write(text)
