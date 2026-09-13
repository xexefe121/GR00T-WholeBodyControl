"""Bind this source-only draft to unchanged producer and independent diagnosis."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
NEW = BASE.parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    source = NEW / 'independent_plant_clock_timeout_correction_v1/source_draft_v1'
    diagnosis = NEW / 'independent_clock_publication_diagnosis_v1/diagnostic_receipt.json'
    assert sha(diagnosis) == 'bbbb8a9d38554ead77a32b9d224647775c517dac518125ebbad68837b6f18975'
    paths = [source / name for name in ('clock_core.py', 'session.py', 'shared_mailbox.py',
                                       'recorded_protocol.py', 'dummy_worker.py', 'run_clock.py')]
    paths += [diagnosis, diagnosis.parent / 'report.json', diagnosis.parent / 'NOTE.md',
              NEW / 'independent_plant_clock_timeout_correction_v1/run/report.json',
              NEW / 'independent_plant_clock_timeout_correction_v1/owner_completion.json',
              BASE / 'NOTE.md', Path(__file__)]
    pins = {path.resolve().as_posix(): sha(path) for path in paths}
    result = dict(kind='source_only_pending_publication_design', implementation_frozen=False,
                  actual_run_selected=False, initial_plus_retry_limit=10,
                  retry_physics_indices_for_activation_420=list(range(4191, 4200)),
                  observed_busy_start_ns=132991903623, observed_busy_end_ns=132991908848,
                  original_activation_deadline_ns=133011372988,
                  slack_after_busy_return_ns=133011372988-132991908848,
                  maximum_jobs=1818, maximum_plant_publication_attempts=18180,
                  maximum_result_poll_records=36380, maximum_transport_records=54560,
                  exact_foundation_event_bound_pending=True,
                  original_timing_failure_preserved=True, worker_result_busy_out_of_scope=True,
                  counterfactual_success_claim=False, input_sha256=pins,
                  native_steps=0, model_calls=0, optimizer_updates=0,
                  task_array_evaluations=0, spawned_workers=0, synthetic_tests_run=0)
    assert all(sha(path) == value for path, value in pins.items())
    with (BASE / 'assessment.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'assessment_sha256': sha(BASE / 'assessment.json'),
                      'note_sha256': sha(BASE / 'NOTE.md'), 'pins': len(pins)}))


if __name__ == '__main__':
    main()
