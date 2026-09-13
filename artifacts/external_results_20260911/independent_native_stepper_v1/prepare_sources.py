"""Copy selected immutable foundations; no native API is imported or called."""
from pathlib import Path
import hashlib,json,shutil
BASE=Path(__file__).resolve().parent;NEW=BASE.parent;SOURCE=BASE/'source_draft_v1'
SOURCE.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
foundation=NEW/'independent_plant_clock_foundation_v1'
receipt=json.loads((foundation/'preparation_report_v3.json').read_text())
pins={}
for name in ('clock_core.py','history.py','mailbox.py'):
    old=foundation/'source_draft_v3'/name
    expected=receipt['source_sha256']['source_draft_v3/'+name]
    assert sha(old)==expected;shutil.copy2(old,SOURCE/name)
    pins[name]=dict(original=str(old),sha256=expected)
oracle=NEW/'bfm_entry250_actual_oracle_v1/source_snapshot_v1/gear_sonic/utils/g1_true23_feasibility_referee.py'
assert sha(oracle)=='0027210cda5a44255debecd7151ecd454e2281a5ff4e3137641d102701fb1330'
obs=NEW/'one_step_physical_student_v1/source_snapshot_v1/gear_sonic/utils/g1_true23_bfm_seed_observations.py'
assert sha(obs)=='a3c68a9aefad87c3dbda5b515f34a968923e9d35234a278b01b727d73062f68d'
for old,name in [(oracle,'oracle_source.py'),(obs,'bfm_observations.py')]:
    shutil.copy2(old,SOURCE/name);pins[name]=dict(original=str(old),sha256=sha(old))
with (BASE/'original_sources.json').open('x') as stream:json.dump(pins,stream,indent=2)
print('Copied five immutable sources; native/model calls zero.')
