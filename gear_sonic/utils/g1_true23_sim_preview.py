"""Simulator publication contract. Does not alter controller or robot APIs."""
import copy
import json
import time

import numpy as np

RUNNER_CONTRACT_VERSION = 'native23-sim-preview-v1'


class SimulatorPreviewRejected(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__('Simulator rejected unacceptable native preview')


def publish_checked(command, diagnostic, context, policy, records, publish):
    """Own evidence before publication; a rejected proposal never reaches publish."""
    if policy not in ('strict', 'diagnostic-only'):
        raise ValueError('Unknown simulator preview failure policy')
    status = command.status.get('native_preview_guard')
    record = None
    if status is not None:
        record = dict(copy.deepcopy(diagnostic or {}), **context,
                      preview_status=copy.deepcopy(status), policy=policy,
                      published=False, application_time_ns=None)
        records.append(record)
        finite = np.isfinite(command.targets).all()
        acceptable = finite and status.get('predicted_limits_satisfied') is True
        record['acceptable'] = bool(acceptable)
        if not acceptable and policy == 'strict':
            record['rejected_at_ns'] = time.monotonic_ns()
            raise SimulatorPreviewRejected(record)
    accepted = publish()
    if record is not None:
        record.update(context)
        record['publish_accepted'] = list(accepted)
        record['published'] = bool(any(accepted))
        record['all_channels_accepted'] = bool(all(accepted))
    return accepted


def stop_simulator(bridge, stop):
    # Stop the independent plant before logging or releasing setup waiters.
    if bridge is not None:
        bridge.lib.clock_stop(bridge.address)
    stamp = time.monotonic_ns()
    stop.set()
    return stamp


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_diagnostics(path, records, timing, epoch_ns):
    """Serialize after plant stop; actual application comes from native trace."""
    first = {}
    for step, row in enumerate(timing):
        first.setdefault(int(row[4]), (step, int(row[2]), int(row[0])))
    with path.open('w', encoding='utf8') as stream:
        for record in records:
            applied = first.get(record['control']) if record['published'] else None
            record['application_time_ns'] = applied[1] if applied else None
            record['application_physics_step'] = applied[0] if applied else None
            record['application_time_kind'] = 'native PD computation start of first application step'
            record['application_deadline_ns'] = applied[2] if applied else None
            if applied:
                record['old_target_substeps'] = max(0, applied[0] - record['control'] * 10)
                record['application_delay_ms'] = (applied[2]-epoch_ns-record['control']*20_000_000)*1e-6
                record['actual_application_delay_ms'] = (applied[1]-epoch_ns-record['control']*20_000_000)*1e-6
            stream.write(json.dumps(json_safe(record), allow_nan=False) + '\n')


def terminal_standing(trailing, trailing30, reached):
    if not reached:
        absent = dict(passed=False, status='not_reached', reason='complete motion and terminal 30-second hold not reached')
        return absent.copy(), absent.copy()
    return dict(trailing, status='evaluated'), dict(trailing30, status='evaluated')
