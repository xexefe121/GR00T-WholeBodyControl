"""Synthetic checks for the independent auditor; no task input evaluation."""
from pathlib import Path
import unittest
import numpy as np
from saved_math import (check_probe, expected_radii, original_difference,
                        duplicate_conflicts, canonical_feature_digest)

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'bfm_entry250_actual_oracle_v1/source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py'

class MathTests(unittest.TestCase):
    def state(self):
        q = np.zeros(30, np.float64)
        q[2], q[3] = .75, 1.
        v = np.zeros(29, np.float64)
        return q, v, np.r_[q, v], np.tile([-1., 1.], (23, 1)), np.full(23, 20.)

    def test_exact_linear_coordinate_change(self):
        q, v, plan, limits, caps = self.state()
        difference = original_difference(CORE)
        radii = expected_radii(caps)
        for axis in list(range(3)) + list(range(6, 58)):
            a, b = q.copy(), v.copy()
            if axis < 3:
                a[axis] += radii[axis]
            elif axis < 29:
                a[axis+1] += radii[axis]
            else:
                b[axis-29] += radii[axis]
            _, change = check_probe(q, v, a, b, axis, radii[axis], plan, difference, limits, caps)
            self.assertAlmostEqual(change[axis], radii[axis])

    def test_unrequested_coordinate_change_rejected(self):
        q, v, plan, limits, caps = self.state()
        a = q.copy()
        a[7] += .01
        a[8] += .001
        with self.assertRaisesRegex(AssertionError, 'coordinate'):
            check_probe(q, v, a, v, 6, .01, plan, original_difference(CORE), limits, caps)

    def test_static_bounds_rejected(self):
        q, v, plan, limits, caps = self.state()
        a = q.copy()
        a[7] = 2.
        with self.assertRaisesRegex(AssertionError, 'position bound'):
            check_probe(q, v, a, v, 6, 2., plan, original_difference(CORE), limits, caps)
        b = v.copy()
        b[6] = 20.
        with self.assertRaisesRegex(AssertionError, 'speed'):
            check_probe(q, v, q, b, 35, 20., plan, original_difference(CORE), limits, caps)

    def test_nonunit_quaternion_rejected(self):
        q, v, plan, limits, caps = self.state()
        a = q.copy()
        a[3] = 1.1
        with self.assertRaisesRegex(AssertionError, 'quaternion'):
            check_probe(q, v, a, v, 3, .01, plan, original_difference(CORE), limits, caps)

    def test_signed_zero_duplicate_and_conflicting_label(self):
        feature = np.zeros((1, 1000), np.float32)
        negative = feature.copy()
        negative[0, 0] = -0.
        index = {}
        self.assertEqual(duplicate_conflicts(feature, np.zeros((1, 23)), np.ones(23), np.zeros(23), index, 'first'), ([], []))
        duplicates, conflicts = duplicate_conflicts(negative, np.ones((1, 23)), np.ones(23), np.zeros(23), index, 'second')
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]['first'], ['first', 0])
        self.assertEqual(conflicts[0]['repeated'], ['second', 0])

    def test_raw_rounding_difference_is_disclosed(self):
        feature = np.zeros((1, 1000), np.float32)
        first = np.ones((1, 23), np.float64)
        second = first.copy()
        second[0, 0] += 1e-12
        index = {}
        duplicate_conflicts(feature, first, np.ones(23), np.zeros(23), index, 'first')
        duplicates, conflicts = duplicate_conflicts(feature, second, np.ones(23), np.zeros(23), index, 'second')
        self.assertFalse(conflicts)
        self.assertGreater(duplicates[0]['raw_target_max_difference'], 0.)

    def test_nonfinite_feature_rejected(self):
        feature = np.zeros(1000, np.float32)
        feature[5] = np.nan
        with self.assertRaises(AssertionError):
            canonical_feature_digest(feature)

if __name__ == '__main__':
    unittest.main()
