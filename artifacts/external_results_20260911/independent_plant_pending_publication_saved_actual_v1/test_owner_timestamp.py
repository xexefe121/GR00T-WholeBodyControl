"""Pure chronology regressions, no completed audit/PID/source checks executed."""
import unittest
from verify_completion_dispatch_v3 import utc_order_key

class TimestampTests(unittest.TestCase):
    def test_actual_seven_digit_timestamps(self):
        self.assertGreater(utc_order_key('2026-09-11T20:09:48.9406372Z'),utc_order_key('2026-09-11T20:09:01.3579158Z'))
        self.assertEqual(utc_order_key('2026-09-11T20:09:48.9406372Z')[-1],940637200)
    def test_exact_hundred_nanosecond_order_same_second(self):
        self.assertLess(utc_order_key('2026-09-11T20:09:48.0000001Z'),utc_order_key('2026-09-11T20:09:48.0000002Z'))
    def test_nine_digit_and_UTC_alias(self):
        self.assertLess(utc_order_key('2026-09-11T20:09:48.000000001Z'),utc_order_key('2026-09-11T20:09:48.000000002Z'))
        self.assertEqual(utc_order_key('2026-09-11T20:09:48.1Z'),utc_order_key('2026-09-11T20:09:48.100000000+00:00'))
        self.assertEqual(utc_order_key('2026-09-11T20:09:48Z')[-1],0)
    def test_calendar_validation(self):
        for value in ('2026-02-29T20:09:48.1Z','2026-13-11T20:09:48.1Z','2026-09-31T20:09:48.1Z','2026-09-11T24:09:48.1Z','2026-09-11T20:09:60.1Z'):
            with self.subTest(value=value),self.assertRaises(ValueError):utc_order_key(value)
        self.assertEqual(utc_order_key('2024-02-29T00:00:00.1Z')[:3],(2024,2,29))
    def test_nonUTC_malformed_fraction_and_types_rejected(self):
        for value in ('2026-09-11T20:09:48.1+01:00','2026-09-11T20:09:48.1-00:00','2026-09-11T20:09:48.1+0000',
            '2026-09-11T20:09:48.Z','2026-09-11T20:09:48.1234567890Z','2026-09-11 20:09:48.1Z','2026-09-11T20:09:48.1Z ',None,1):
            with self.subTest(value=value),self.assertRaises(ValueError):utc_order_key(value)

if __name__=='__main__':unittest.main()
