"""Pure saved-array/stub tests. No graph session construction or native dynamics."""
import ast,importlib.util,json,tempfile,types,sys,unittest
from pathlib import Path
import numpy as np
import collection_arrays as arrays
from branch_inputs import row_mapping,advance_history,clock_table,fixed_map_function,archive,exact
from stateless_adapter import definitions

NEW=Path(__file__).resolve().parents[2]
class CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.centers=archive(NEW/'velocity_chord_student_v1/generation/centers.npz')
    def test_row_mapping_and_successor_clock(self):
        rows=row_mapping();self.assertEqual(len(rows),3054);self.assertEqual(len(set(rows)),3054)
        self.assertEqual(rows[2036],(2,250,2038));self.assertEqual(rows[-1],(2,1267,3055))
        for d,c,i in rows:
            self.assertEqual(int(self.centers['control'][i]),c);self.assertEqual(int(self.centers['control'][i+1]),c+1)
            self.assertEqual(int(self.centers['source_frame'][i+1]),c+12)
    def test_history_matches_original_newest_first_without_mutation(self):
        scope=definitions(Path(__file__).parent/'frozen/g1_true23_bfm_seed_observations.py',['BFMHistory'],dict(np=np))
        for index in (0,1017,1019,2036,2038,3055):
            c=self.centers;state=c['state'][index];prior=c['previous_action'][index]
            h=scope['BFMHistory']();h.data={key:c['history_'+key][index].copy() for key in arrays.HISTORY_WIDTHS}
            before={key:value.copy() for key,value in h.data.items()}
            terms=dict(actions=prior,base_ang_vel=state[49:52],dof_pos=state[:23],dof_vel=state[23:46],projected_gravity=state[46:49])
            original_before=h.before_update(terms);named,flat=advance_history(c,index)
            exact(original_before,c['history'][index],'old lag input')
            for key in named:
                exact(named[key],h.data[key],'advanced original '+key)
                exact(before[key],c['history_'+key][index],'center unchanged '+key)
                exact(named[key][0],terms[key],'newest sample '+key)
            exact(flat,np.concatenate([h.data[key].ravel() for key in sorted(h.data)]),'flat history')
    def test_calibration_history_matches_actual251(self):
        actual=archive(NEW/'velocity_chord_student_evaluation_v1/nominal/trace.npz')
        _,flat=advance_history(self.centers,2038);exact(flat,actual['history'][251],'calibration history')
    def test_all_nominal_full_fixed_maps_exact(self):
        function=fixed_map_function();c=self.centers
        for i in range(3057):
            result=function(c,i,c['qpos'][i],c['qvel'][i],c['joint_limits'])
            for key,source in [('label_fixed_map_target','expert_target'),('teacher_feedback_raw','teacher_feedback_raw'),('teacher_feedback_correction','teacher_feedback_correction'),('teacher_preclip_target','teacher_preclip')]:exact(result[key],c[source][i],key)
    def test_clock_is_repeated_addition(self):
        values=clock_table();self.assertEqual(len(values),12681)
        self.assertTrue(np.array_equal(values[1:],values[:-1]+.002));self.assertNotEqual(values[2500],5.)
    def test_empty_evidence_schema_manifest(self):
        old=arrays.ROWS;arrays.ROWS=2
        try:
            with tempfile.TemporaryDirectory() as folder:
                s=arrays.Store(Path(folder)/'data')
                for phase in ('nominal','policy'):
                    self.assertEqual(s.arrays[phase+'_qpos'].shape,(2,11,30));self.assertEqual(s.arrays[phase+'_warning_counts'].dtype,np.int32)
                    self.assertTrue(np.isnan(s.arrays[phase+'_qpos']).all());self.assertFalse(s.arrays['label_valid'].any())
                s.row(0,policy_status=2,policy_valid_steps=3)
                self.assertTrue(np.isnan(s.arrays['endpoint_features'][0]).all())
                manifest=s.manifest(dict(complete=False))
                for name,spec in manifest['arrays'].items():
                    self.assertEqual(arrays.sha(s.folder/spec['path']),spec['sha256']);self.assertEqual(list(s.arrays[name].shape),spec['shape'])
                for value in s.arrays.values():value._mmap.close()
        finally:arrays.ROWS=old
    def test_native_step_attempt_return_capture_and_exception(self):
        fake_mj=types.ModuleType('mujoco');fake_forecast=types.ModuleType('native_forecast');fake_forecast.NativeForecast=object
        saved={k:sys.modules.get(k) for k in ('mujoco','native_forecast')};sys.modules.update(mujoco=fake_mj,native_forecast=fake_forecast)
        try:
            spec=importlib.util.spec_from_file_location('capture_stub_under_test',Path(__file__).parent/'native_capture.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            obj=module.CapturedForecast.__new__(module.CapturedForecast)
            d=types.SimpleNamespace(qpos=np.arange(30,dtype=float),qvel=np.arange(29,dtype=float),time=.002,ctrl=np.arange(23,dtype=float),qfrc_actuator=np.arange(29,dtype=float),warning=types.SimpleNamespace(number=np.zeros(8,np.int32),lastinfo=np.zeros(8,np.int32)))
            obj.model=object();obj.engine=types.SimpleNamespace(private=d);obj.active=True
            obj.attempted=obj.returned=obj.captured=obj.total_attempted=obj.total_returned=0;obj.expected=0.
            obj.q=np.full((11,30),np.nan);obj.v=np.full((11,29),np.nan);obj.t=np.full(11,np.nan);obj.et=np.full(11,np.nan)
            obj.warnings=np.zeros((11,8),np.int32);obj.lastinfo=np.zeros((11,8),np.int32);obj.torque=np.full((10,23),np.nan);obj.force=np.full((10,23),np.nan)
            obj.native_step=lambda m,data:None;obj._step(obj.model,d)
            self.assertEqual((obj.attempted,obj.returned,obj.captured),(1,1,1));exact(obj.force[0],d.qfrc_actuator[6:],'captured force')
            def fail(m,data):raise RuntimeError('fake native partial failure')
            obj.native_step=fail
            with self.assertRaises(RuntimeError):obj._step(obj.model,d)
            self.assertEqual((obj.attempted,obj.returned,obj.captured),(2,1,1));self.assertTrue(np.isnan(obj.q[2]).all())
            obj.active=False
            with self.assertRaises(RuntimeError):obj._step(obj.model,d)
            self.assertEqual(obj.attempted,2)
        finally:
            for key,value in saved.items():
                if value is None:sys.modules.pop(key,None)
                else:sys.modules[key]=value
    def test_no_optimization_and_only_counted_session_calls(self):
        text=(Path(__file__).parent/'collect_branches.py').read_text();tree=ast.parse(text)
        self.assertNotIn('optimizer',','.join(n.id for n in ast.walk(tree) if isinstance(n,ast.Name)))
        self.assertEqual(text.count('.session.run('),1)
    def test_graph_failure_clears_previous_return_identity(self):
        import collect_branches as runtime
        old_dest=runtime.DEST;old_progress=runtime.PROGRESS.copy();old_counts={k:v.copy() for k,v in runtime.COUNTS.items()}
        try:
            with tempfile.TemporaryDirectory() as folder:
                runtime.DEST=Path(folder);runtime.PROGRESS['nominal_verified']=3054;runtime.PROGRESS['active_row']=2036
                runtime.ACTIVE.clear();runtime.ACTIVE['head_output_0']=np.ones((1,23),np.float32)
                class Failed:
                    def run(self,*args):raise RuntimeError('stub graph exception')
                with self.assertRaises(RuntimeError):runtime.Counted(Failed(),'head').run(None,{'features':np.zeros((1,1069),np.float32)})
                self.assertNotIn('head_output_0',runtime.ACTIVE);self.assertFalse(runtime.ACTIVE['active_graph_returned'])
                self.assertEqual(runtime.COUNTS['head'],dict(attempted=1,returned=0))
                saved=archive(runtime.DEST/'active.npz');self.assertNotIn('head_output_0',saved);self.assertFalse(saved['active_graph_returned'])
        finally:
            runtime.DEST=old_dest;runtime.PROGRESS.clear();runtime.PROGRESS.update(old_progress)
            for k,v in old_counts.items():runtime.COUNTS[k].update(v)
            runtime.ACTIVE.clear()

if __name__=='__main__':unittest.main(verbosity=2)
