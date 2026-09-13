import unittest
import numpy as np
from feature_scan import Groups, reduced


class TestScan(unittest.TestCase):
    def rows(self):
        return [(np.zeros(1069, dtype=np.float32), np.zeros(23, dtype=np.float64)) for _ in range(2)]

    def scan(self, rows, **kwargs):
        groups = Groups(lambda i: rows[i], **kwargs)
        for i, (x, y) in enumerate(rows): groups.add(i, x, y)
        return groups

    def test_absolute_conflict(self):
        rows = self.rows(); rows[1][1][0] = .2
        self.assertEqual(self.scan(rows).target_numeric_conflicts, [[0, 1]])

    def test_removed_features_do_not_distinguish(self):
        rows = self.rows(); rows[1][0][52:75] = 2; rows[1][0][1023:] = -3
        self.assertEqual(self.scan(rows).duplicates, 1)

    def test_kept_features_distinguish(self):
        for index in [0, 51, 75, 1022]:
            rows = self.rows(); rows[1][0][index] = 1
            self.assertEqual(self.scan(rows).duplicates, 0)

    def test_signed_zero_preserved(self):
        rows = self.rows(); rows[1][0][0] = -0.0
        self.assertEqual(self.scan(rows).duplicates, 0)
        self.assertEqual(self.scan(rows, numeric_zero=True).duplicates, 1)
        self.assertTrue(np.signbit(rows[1][0][0]))

    def test_target_signed_zero_separate(self):
        rows = self.rows(); rows[1][1][0] = -0.0
        result = self.scan(rows)
        self.assertEqual(result.target_byte_conflicts, [[0, 1]])
        self.assertEqual(result.target_numeric_conflicts, [])

    def test_hash_collision_requires_actual_bytes(self):
        rows = self.rows(); rows[1][0][0] = 1
        result = self.scan(rows, digest=lambda _: b'x')
        self.assertEqual(result.duplicates, 0)
        self.assertEqual(result.hash_collisions, 1)

    def test_dtype_rejected(self):
        with self.assertRaises(ValueError): reduced(np.zeros(1069, dtype=np.float64))
        with self.assertRaises(ValueError): self.scan([(np.zeros(1069, np.float32), np.zeros(23, np.float32))])

    def test_nonfinite_rejected(self):
        rows = self.rows(); rows[1][0][0] = np.nan
        with self.assertRaises(ValueError): self.scan(rows)


if __name__ == '__main__': unittest.main()
