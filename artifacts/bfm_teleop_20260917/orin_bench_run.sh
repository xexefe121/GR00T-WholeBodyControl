#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/unitree/bfm_orin_bench
export PYTHONPATH="$ROOT/src"
PY="$ROOT/venv/bin/python"
"$PY" "$ROOT/src/orin_bench.py" c1 --root "$ROOT" --output "$ROOT/reports/c1.json"
"$PY" - "$ROOT/reports" <<'PY'
import json, sys
report=json.load(open(sys.argv[1]+'/c1.json'))
if not report['passed']:
    raise SystemExit('C1 failed; B1 and B2 intentionally not run')
PY
for threads in 1 2 4 6; do
  "$PY" "$ROOT/src/orin_bench.py" b1 --threads "$threads" --root "$ROOT" --output "$ROOT/reports/b1_${threads}.json"
done
"$PY" - "$ROOT/reports" <<'PY'
import json, sys
root=sys.argv[1]
rows=[json.load(open(f'{root}/b1_{n}.json')) for n in (1,2,4,6)]
best=min(rows,key=lambda x:x['components']['total_ms']['p99'])['threads']
json.dump({'best_threads':best,'criterion':'lowest B1 total p99'},open(f'{root}/selection.json','w'),indent=2)
PY
best=$("$PY" -c "import json; print(json.load(open('$ROOT/reports/selection.json'))['best_threads'])")
for run in 1 2; do
  "$PY" "$ROOT/src/orin_bench.py" b2 --threads "$best" --root "$ROOT" --output "$ROOT/reports/b2_${run}.json"
done
"$PY" - "$ROOT/reports" <<'PY'
import json, sys
root=sys.argv[1]; best=json.load(open(root+'/selection.json'))['best_threads']
one,two=(json.load(open(root+f'/b2_{i}.json')) for i in (1,2))
json.dump({'best_threads':best,'runs':[one,two],'feasible':all(r['deadline_misses']==0 and r['work_ms']['p99']<15.0 for r in (one,two))},open(root+'/summary.json','w'),indent=2)
PY
