"""Source/stub composition review only; no native/model/worker execution."""
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
BASE = OUT.parent/'independent_plant_process_clock_v1'
SOURCE = BASE/'source_draft_v3'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
pre = json.loads((BASE/'source_preparation_v3.json').read_text())
assert sha(BASE/'source_preparation_v3.json') == 'a1c551b6a868ffe9d44db30bc7848d25d9121fe156a290ec706bfbd04ba7e513'
assert pre['source_preparation_passed'] and pre['all_input_hashes_unchanged']
for path, expected in pre['input_sha256'].items():
    assert sha(path) == expected, path
assert len(pre['input_sha256']) == 62
sys.path.insert(0, str(SOURCE))
suite = unittest.defaultTestLoader.discover(str(SOURCE), pattern='test_scaffold.py')
capture = io.StringIO()
result = unittest.TextTestRunner(stream=capture, verbosity=2).run(suite)
(OUT/'synthetic_tests.txt').write_text(capture.getvalue())
assert result.testsRun == 20 and result.wasSuccessful() and not result.skipped
for path, expected in pre['input_sha256'].items():
    assert sha(path) == expected, path
report = dict(source_review_pass=True, reviewed_utc=datetime.now(timezone.utc).isoformat(),
    preparation_sha256=sha(BASE/'source_preparation_v3.json'),
    input_pins_verified=62, input_sha256=pre['input_sha256'],
    source_review_sha256=sha(__file__), synthetic_tests_passed=result.testsRun,
    tests_log_sha256=sha(OUT/'synthetic_tests.txt'),
    reviewed_contracts=[
        'Recorded table retains original raw feedback separately from bounded targets.',
        'One-control-future jobs bind measured snapshot, source window, history and predecessor action.',
        'Unchanged foundation owns actual admission timestamp and holds on latched missing input.',
        'Independent component predicate additionally rejects held commands, wrong coverage and outer deadline misses.',
        'Full291 initialization and373-f64 captures retained; continuous hold uses same live instance.',
        'Owned exporter preserves full captured failing step and uncommitted overflow evidence.',
        'Failed worker start never joins an unstarted handle or claims process absence.',
        'Same epoch deadlines, debt abort and source clocks remain unchanged.'
    ],
    preparation_selected='Concrete runtime loader/counting wrapper, input freeze, durable supervisor and postrun writer may now be prepared with stub tests.',
    actual_benchmark_selected=False, actual_process_clock_runs=0, actual_worker_processes=0,
    model_calls=0, native_steps=0, optimizer_updates=0,
    remaining_before_actual_run=[
        'Exact runtime/import and original trace/fixture/model input freeze with verified loader/API accounting.',
        'Supervisor cleanup and known worker failure/exit evidence including uncertain failed-start paths.',
        'Fixed epoch admission only after readiness/preallocation and no disk or hash inventory during epoch.',
        'Concrete whole18190step request and saved-output schema/independent comparison request.',
        'Final review of actual runner/request before one selected benchmark launch.'
    ],
    limitation='Recorded-command scheduling/transport component only, with conservative pre-wait polling. No online-policy or hard-real-time qualification.')
(OUT/'review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(passed=True, tests=result.testsRun, review_sha256=sha(OUT/'review.json'))))
