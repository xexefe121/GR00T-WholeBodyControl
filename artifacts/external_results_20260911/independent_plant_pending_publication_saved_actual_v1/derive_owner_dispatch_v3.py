"""Chronology parser correction only; v2 and original verifier remain intact."""
import difflib,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
OLD=BASE/'verify_completion_dispatch_v2.py';NEW=BASE/'verify_completion_dispatch_v3.py'
PARSER='''
def utc_order_key(value):
    """Strict UTC timestamps with optional 1..9 fractional digits, lossless ns."""
    if type(value) is not str:raise ValueError('UTC timestamp must be text')
    match=re.fullmatch(r'(\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2})(?:\\.(\\d{1,9}))?(?:Z|\\+00:00)',value)
    if match is None:raise ValueError('Strict UTC Z/+00:00 timestamp required')
    whole=datetime.strptime(match.group(1),'%Y-%m-%dT%H:%M:%S')
    nanosecond=int((match.group(2) or '').ljust(9,'0'))
    return (whole.year,whole.month,whole.day,whole.hour,whole.minute,whole.second,nanosecond)

'''
old=OLD.read_text();text=old.replace('import hashlib,json','import hashlib,json,re')
assert text.count('def validate_dispatch_correction(dispatch):')==1
text=text.replace('def validate_dispatch_correction(dispatch):',PARSER+'def validate_dispatch_correction(dispatch):')
before="datetime.fromisoformat(dispatch['utc'].replace('Z','+00:00'))>datetime.fromisoformat(stop['utc'].replace('Z','+00:00'))"
assert text.count(before)==1
text=text.replace(before,"utc_order_key(dispatch['utc'])>utc_order_key(stop['utc'])")
text=text.replace("BASE/'owner_completion_dispatch_v2.json'","BASE/'owner_completion_dispatch_v3.json'")
marker="    output_paths += [BASE/'dispatch.json'"
assert text.count(marker)==1
text=text.replace(marker,"    output_paths += [BASE/'verify_completion_dispatch_v2.py',BASE/'derive_owner_dispatch_v3.py',BASE/'owner_dispatch_v3.diff',BASE/'test_owner_timestamp.py',BASE/'owner_timestamp_tests.log']\n"+marker)
with NEW.open('x',encoding='utf-8') as f:f.write(text)
with (BASE/'owner_dispatch_v3.diff').open('x') as f:f.write(''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='preserved/verify_completion_dispatch_v2.py',tofile='verify_completion_dispatch_v3.py')))
print(json.dumps(dict(source_sha256=hashlib.sha256(NEW.read_bytes()).hexdigest(),preserved_v2_sha256=hashlib.sha256(OLD.read_bytes()).hexdigest(),executed=False)))
