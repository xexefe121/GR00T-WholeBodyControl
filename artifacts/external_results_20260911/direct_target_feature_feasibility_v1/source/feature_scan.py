"""Pure saved-array grouping; neither features nor teacher labels are produced."""
import hashlib
import numpy as np


def reduced(row, numeric_zero=False):
    if row.dtype != np.dtype('float32') or row.shape != (1069,):
        raise ValueError('expected original float32[1069]')
    result = np.concatenate((row[:52], row[75:1023]))
    if not np.isfinite(result).all():
        raise ValueError('nonfinite retained feature')
    if numeric_zero:
        # Secondary equivalence key only. Original arrays and primary keys stay exact.
        result[result == 0] = np.float32(0)
    return result


class Groups:
    def __init__(self, row_reader, numeric_zero=False, digest=None):
        self.row_reader = row_reader
        self.numeric_zero = numeric_zero
        self.digest = digest or (lambda value: hashlib.sha256(value).digest())
        self.buckets = {}
        self.representatives = []
        self.counts = []
        self.duplicates = 0
        self.target_byte_conflicts = []
        self.target_numeric_conflicts = []
        self.hash_collisions = 0

    def add(self, row_id, features, target):
        if target.dtype != np.dtype('float64') or target.shape != (23,) or not np.isfinite(target).all():
            raise ValueError('expected finite original float64[23] absolute target')
        key = reduced(features, self.numeric_zero).tobytes()
        digest = self.digest(key)
        bucket = self.buckets.setdefault(digest, [])
        for group in bucket:
            prior_id = self.representatives[group]
            prior_features, prior_target = self.row_reader(prior_id)
            if reduced(prior_features, self.numeric_zero).tobytes() != key:
                self.hash_collisions += 1
                continue
            self.counts[group] += 1
            self.duplicates += 1
            if prior_target.tobytes() != target.tobytes():
                pair = [prior_id, row_id]
                self.target_byte_conflicts.append(pair)
                if not np.array_equal(prior_target, target):
                    self.target_numeric_conflicts.append(pair)
            return digest, group
        group = len(self.representatives)
        self.representatives.append(row_id)
        self.counts.append(1)
        bucket.append(group)
        return digest, group

    def summary(self):
        return dict(rows=sum(self.counts), unique_inputs=len(self.counts),
                    duplicate_rows=self.duplicates, duplicate_groups=sum(c > 1 for c in self.counts),
                    largest_group=max(self.counts, default=0),
                    absolute_target_byte_conflict_pairs=len(self.target_byte_conflicts),
                    absolute_target_numeric_conflict_pairs=len(self.target_numeric_conflicts),
                    hash_collision_comparisons=self.hash_collisions)
