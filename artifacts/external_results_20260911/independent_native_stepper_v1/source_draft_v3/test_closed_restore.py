"""Closed adapter must reject initialization before state writes; fake API only."""
import unittest
from test_native_stub import fixture
from native_stepper import NativeAdapterError

class ClosedRestoreTests(unittest.TestCase):
    def test_closed_uninitialized_adapter_rejects_restore_without_mutation(self):
        s,m,d,a=fixture(False)
        initial=d.integration.copy()
        s.verify_model_exit()
        with self.assertRaises(NativeAdapterError):
            s.restore_initial(initial,d.warning.number.copy(),d.warning.lastinfo.copy())
        self.assertTrue(s.closed)
        self.assertFalse(s.initialized)
        self.assertFalse(s.restore_attempted)
        self.assertEqual(a.set_calls,0)
        self.assertEqual(a.calls,[])
        self.assertEqual(d.integration.tobytes(),initial.tobytes())

if __name__=='__main__':unittest.main()
