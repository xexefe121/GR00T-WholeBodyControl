import tempfile,unittest
from pathlib import Path
import numpy as np
from input_schema import archive_headers,load_numeric

class SchemaTests(unittest.TestCase):
    def test_unused_object_metadata_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'bad.npz';np.savez(path,used=np.zeros(2),unused=np.array([object()],dtype=object))
            with self.assertRaisesRegex(ValueError,'object dtype'):load_numeric(path,{'used':((2,),'float64')})
    def test_only_declared_numeric_keys_loaded(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'good.npz';np.savez(path,used=np.zeros(2),unused=np.ones(3))
            values,headers=load_numeric(path,{'used':((2,),'float64')})
            self.assertEqual(list(values),['used']);self.assertEqual(set(headers),{'used','unused'})
    def test_consumed_shape_dtype_and_finiteness_fail(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'bad.npz';np.savez(path,used=np.array([np.nan]))
            with self.assertRaisesRegex(ValueError,'schema'):load_numeric(path,{'used':((2,),'float64')})
            with self.assertRaisesRegex(ValueError,'nonfinite'):load_numeric(path,{'used':((1,),'float64')})

if __name__=='__main__':unittest.main()
