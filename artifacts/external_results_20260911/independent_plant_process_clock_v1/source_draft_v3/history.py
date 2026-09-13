"""Plant-owned pure BFM four-lag operation, with immutable byte snapshots."""
import numpy as np

SIZES = dict(actions=23, base_ang_vel=3, dof_pos=23, dof_vel=23, projected_gravity=3)


def freeze_named(data):
    return tuple((key, np.asarray(data[key], dtype=np.float32).tobytes()) for key in sorted(data))


class MeasuredHistory:
    def __init__(self):
        self._data = {key: np.zeros((4, size), np.float32) for key, size in SIZES.items()}
        self.entries = 0

    def snapshot(self):
        return freeze_named(self._data)

    def advance(self, measured_terms, incoming_raw):
        # Validate the complete copy before committing any history entry.
        if set(measured_terms) != set(SIZES) - {'actions'}:
            raise ValueError('complete measured terms required; actions belong to plant')
        terms = {key: np.asarray(value, dtype=np.float32).copy()
                 for key, value in measured_terms.items()}
        terms['actions'] = np.frombuffer(incoming_raw, dtype=np.float32).copy()
        for key, size in SIZES.items():
            if terms[key].shape != (size,) or not np.isfinite(terms[key]).all():
                raise ValueError('invalid measured history term: ' + key)
        before = self.snapshot()
        flat = np.concatenate([self._data[key].reshape(-1) for key in sorted(self._data)]).copy()
        # Byte-for-byte arithmetic/order of frozen BFMHistory.before_update.
        for key in self._data:
            self._data[key][1:] = self._data[key][:-1].copy()
            self._data[key][0] = terms[key]
        self.entries += 1
        return before, flat.tobytes(), freeze_named(terms), self.snapshot()
