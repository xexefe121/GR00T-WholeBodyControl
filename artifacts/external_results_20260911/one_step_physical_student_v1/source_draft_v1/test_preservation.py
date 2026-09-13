"""Pure stub preservation checks; no BFM/model/physics calls."""
from pathlib import Path
import tempfile
import numpy as np
import generate_velocity_chords as g
from chord_common import BASE,write_atomic,read,write,sha

def main():
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder);g.DEST=path
        m=np.lib.format.open_memmap(path/'dummy.npy',mode='w+',dtype=np.float64,shape=(2,23))
        m[0]=np.arange(23);m[1]=100
        active=dict(stage=np.asarray('PROBE'),flat_probe_index=np.int64(1),returned_base_target=np.full(23,np.nan),
            qpos=np.arange(30,dtype=np.float64),qvel=np.arange(29,dtype=np.float64),
            previous_action=np.zeros(23,np.float32),history=np.zeros(300,np.float32),goal_latent=np.zeros((1,256),np.float32))
        progress=dict(centers_completed=3057,probes_completed=1)
        failure=g.save_failure(active,{'dummy':m},progress,AssertionError('stub parity failure'))
        assert failure['active_probe_committed'] is False and failure['first_uncommitted_probe_index']==1
        saved=np.load(path/'failure_active.npz');assert np.isnan(saved['returned_base_target']).all()
        np.testing.assert_array_equal(np.load(path/'dummy.npy')[0],np.arange(23))
        write_atomic(path/'progress.json',dict(stage='FAILED',failure=failure))
        assert read(path/'progress.json')['failure']['completed_probes']==1
        assert not (path/'progress.json.tmp').exists()
        progress['probes_completed']=2
        assert g.save_failure(active,{'dummy':m},progress,AssertionError('final hash failure'))['active_probe_committed'] is True
        del m,saved
    output=BASE/'generation_preservation_tests_v2.json';assert not output.exists()
    write(output,dict(passed=True,partial_memmaps_flushed=True,active_nonfinite_output_preserved=True,
        committed_prefix_excludes_active_failure=True,postcompletion_failure_keeps_active_committed=True,
        atomic_progress_replace=True,actual_model_calls=0,physics_steps=0,optimizer_updates=0,
        source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))
    print('PURE PRESERVATION TESTS PASS')

if __name__=='__main__':main()
