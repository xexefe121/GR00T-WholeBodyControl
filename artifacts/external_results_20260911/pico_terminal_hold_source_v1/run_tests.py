from pathlib import Path
import sys

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here / "repo"))
import pytest

raise SystemExit(pytest.main([str(here / "test_continuity.py"), "-q", "--rootdir=" + str(here),
                             "--confcutdir=" + str(here), "--junitxml=" + str(here / "focused_tests_v3.xml")]))
