"""Preserve wrong-host test collection; prepare unchanged-source tests in existing SciPy Python."""
from pathlib import Path
BASE=Path(__file__).resolve().parent
destination=BASE.parent/'direct_target_width512_evaluation_independent_review_v2'
destination.mkdir(exist_ok=False)
text=(BASE/'review_sources.py').read_text()
text=text.replace("pins={PREP.as_posix():EXPECTED,old_prep_path.as_posix():sha(old_prep_path),**prep['input_sha256']}",
    "pins={PREP.as_posix():EXPECTED,old_prep_path.as_posix():sha(old_prep_path),**prep['input_sha256']}\n"
    "    prior=NEW/'direct_target_width512_evaluation_independent_review_v1'\n"
    "    for name in ('review_sources.py','test_exit.json','synthetic_tests.xml','synthetic_tests.log','prepare_scipy_review.py'):\n"
    "        pins[(prior/name).as_posix()]=sha(prior/name)")
text=text.replace("findings=[],reviewed_contracts=[", "findings=[],prior_test_collection_failure='Isolated training Python lacks SciPy; unchanged suite rerun in existing SciPy CPU Python. No source modification or package installation.',reviewed_contracts=[")
with (destination/'review_sources.py').open('x',encoding='utf-8') as stream:stream.write(text)
