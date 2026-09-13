"""Bind final clock metadata, representability notes, and immutable scan receipt."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent.parent
NEW = HERE.parent
DATA = NEW/'one_step_policy_branch_collection_resume2969_v1/collection/data'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


request = json.loads((HERE/'request.json').read_text())
report = json.loads((HERE/'results/report.json').read_text())
pins = dict(request['input_source_sha256'])
for path in [Path(__file__), HERE/'request.json', HERE/'results/report.json', HERE/'README.md']:
    pins[str(path.resolve())] = sha(path)
manifest = json.loads((DATA/'manifest.json').read_text())
starts = np.tile(np.arange(250,1268,dtype=np.int64),3)
metadata = {}
for key, expected in [('source_frame',starts+11), ('successor_frame',starts+12)]:
    spec = manifest['arrays'][key]; path = DATA/spec['path']
    assert sha(path) == spec['sha256']
    pins[str(path.resolve())] = spec['sha256']
    value = np.load(path,allow_pickle=False)
    assert value.dtype == expected.dtype and value.tobytes() == expected.tobytes()
    metadata[key] = dict(rows=len(value), byte_exact=True, first=int(value[0]), last=int(value[-1]))
for path, expected in pins.items(): assert sha(path) == expected, path
receipt = dict(kind='direct_target_feature_feasibility_completion', completed=True,
    scan_report_sha256=sha(HERE/'results/report.json'), request_sha256=sha(HERE/'request.json'),
    source_frame_checks=metadata, all_final_pins_exact=True, pin_count=len(pins), pins=pins,
    mandatory_rows=146733, expanded_rows=153580, absolute_target_conflicts=0,
    inference_calls=0, native_steps=0, optimizer_updates=0, new_labels=0)
with (HERE/'completion.json').open('x') as stream: json.dump(receipt,stream,indent=2)
print('completion',sha(HERE/'completion.json'))
