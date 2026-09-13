"""Independent synthetic transport review; no native/controller imports or calls."""
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import unittest

BASE = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
SOURCE = BASE / 'independent_plant_mailbox_transport_v1/source_snapshot_v1'
OUT = Path(__file__).parent
sys.path.insert(0, str(SOURCE))
from shared_mailbox import SharedMailbox, HEADER, EMPTY, FULL, MAX_U64, ZERO_DIGEST
import test_transport

class BoundaryTests(unittest.TestCase):
    def box(self):
        return SharedMailbox.create(mp.get_context('spawn'), 1, 64, b'R' * 32)

    def test_version_exhaustion_after_take(self):
        box = self.box()
        with box._layout.locks[0]:
            box._store_header(0, EMPTY, 0, MAX_U64 - 1, 0, ZERO_DIGEST)
        self.assertEqual(box.try_publish(0, b'last'), 'PUBLISHED')
        item = box.poll_once()[0][2]
        self.assertEqual(item.version, MAX_U64)
        self.assertEqual(box.try_publish(1, b'overflow'), 'VERSION_EXHAUSTED')
        self.assertEqual(box.poll_once()[0][1], 'EMPTY')

    def test_malformed_full_header_never_exposes_payload(self):
        for field, value in [(0, b'badmagic'), (1, 2), (2, 99), (4, 0),
                             (5, 65), (6, b'W' * 32)]:
            box = self.box()
            self.assertEqual(box.try_publish(0, b'complete'), 'PUBLISHED')
            with box._layout.locks[0]:
                header = list(box._load_header(0))
                header[field] = value
                view = box._view()
                try:
                    HEADER.pack_into(view, 0, *header)
                finally:
                    view.release()
            self.assertEqual(box.poll_once(), ((0, 'CORRUPT', None),))
            self.assertEqual(box.try_publish(0, b'overwrite'), 'CORRUPT')

    def test_empty_payload_is_committed_immutable_publication(self):
        box = self.box()
        self.assertEqual(box.try_publish(MAX_U64, b''), 'PUBLISHED')
        item = box.poll_once()[0][2]
        self.assertEqual((item.key, item.version, item.payload), (MAX_U64, 1, b''))
        self.assertEqual(box.poll_once(), ((0, 'EMPTY', None),))

    def test_close_is_endpoint_local_and_preserves_pending_record(self):
        box = self.box()
        peer = box.endpoint(b'R' * 32)
        self.assertEqual(box.try_publish(0, b'pending'), 'PUBLISHED')
        box.close()
        self.assertEqual(box.poll_once(), ((0, 'CLOSED', None),))
        self.assertEqual(peer.poll_once()[0][2].payload, b'pending')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if __name__ == '__main__':
    receipt_path = BASE / 'independent_plant_mailbox_transport_v1/source_test_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    before = {str(SOURCE / name): digest for name, digest in receipt['source_sha256'].items()}
    before.update(receipt['input_sha256'])
    assert all(sha(Path(path)) == digest for path, digest in before.items())
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromModule(test_transport),
        unittest.defaultTestLoader.loadTestsFromTestCase(BoundaryTests),
    ])
    with (OUT / 'tests.log').open('w') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    unchanged = all(sha(Path(path)) == digest for path, digest in before.items())
    report = {'source_review_passed': result.wasSuccessful() and unchanged,
              'tests_run': result.testsRun, 'failures': len(result.failures),
              'errors': len(result.errors), 'all_subjects_unchanged': unchanged,
              'input_sha256': before, 'source_test_receipt_sha256': sha(receipt_path),
              'review_source_sha256': sha(Path(__file__)),
              'tests_log_sha256': sha(OUT / 'tests.log'),
              'native_steps': 0, 'task_model_calls': 0, 'optimizer_updates': 0,
              'plant_ticks': 0, 'actual_clock_runs': 0,
              'real_time_qualified': False,
              'limitations': ['Windows spawn only; Linux validation still required.',
                 'Dead owner can leave permanent BUSY slot; no epoch-local recovery.',
                 'Copying, hashing, allocation and scheduling are not bounded by lock semantics.',
                 'Existing foundation admits TAKEN only; future integration must log added statuses.',
                 'Use an isolated module namespace when integrating; local mailbox shadows stdlib.']}
    (OUT / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['source_review_passed', 'tests_run', 'failures', 'errors']}))
    raise SystemExit(0 if report['source_review_passed'] else 1)
