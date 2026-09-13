"""Add the reviewed saved-prefix receipt before any v2 parity execution."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


request_path = HERE / 'request.json'
original = request_path.read_bytes()
assert sha(request_path) == '73cbdf0d00dfabccccfc26eb4ab65450e98cd8b9e60998a758c661c5d304617a'
assert not (HERE/'results').exists() and not (HERE/'request_original.json').exists()
(HERE/'request_original.json').write_bytes(original)
request = json.loads(original)
parent = HERE.parent/'phase_student_preallocated_forecast_v2/results/report.json'
assert sha(parent) == '67e516f3cfd059438677580b8a7b575b183af270327f3143a13748b62201e478'
for path in (parent, Path(__file__), HERE/'request_original.json'):
    request['input_sha256'][str(path)] = sha(path)
request['preexecution_pin_update'] = 'Bind saved232 prefix-comparison report, as independent source review requested; original manifest retained. No v2 diagnostic has run.'
request_path.write_text(json.dumps(request,indent=2)+'\n')
print(json.dumps(dict(request_sha256=sha(request_path), source_and_native_library_unchanged=True)))
