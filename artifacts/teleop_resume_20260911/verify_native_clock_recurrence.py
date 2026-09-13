"""Fast native clock regression: real MuJoCo steps, virtual wall-clock waits.

This checks the production clock guard and fault detection, not controller
behavior or deadline performance. Production scheduling is left unchanged.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import mujoco

root = Path(__file__).resolve().parents[2]
out = Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/native_clock_recurrence_v1')
out.mkdir(parents=True, exist_ok=True)
assert mujoco.__version__ == '3.2.3'
package = Path(mujoco.__file__).parent
source = root / 'gear_sonic/native/true23_clock.cpp'
library = next(package.glob('libmujoco.so*'))
text = source.read_text()
assert 'expected_time+=.002;' in text
assert 'std::abs(d->time-expected_time)>1e-10' in text
assert 'deadline=epoch+int64_t(step+1)*2000000' in text
harness = r'''
#include <mujoco/mujoco.h>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cerrno>
#include <initializer_list>
#include <new>
#include <time.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
static int64_t test_time=0;
static int test_step=0, test_fault=0, old_first_rejection=0;
static int virtual_gettime(clockid_t, timespec *t) {
  t->tv_sec=test_time/1000000000; t->tv_nsec=test_time%1000000000; return 0;
}
static int virtual_nanosleep(clockid_t, int, const timespec *t, timespec *) {
  test_time=int64_t(t->tv_sec)*1000000000+t->tv_nsec; return 0;
}
static void checked_step(const mjModel *m, mjData *d) {
  mj_step(m,d); ++test_step;
  if(test_step==100) {
    if(test_fault==1)d->time=0;
    if(test_fault==2)d->time-=.002;
    if(test_fault==3)d->time+=.002;
  }
  if(test_fault==4)d->time+=2e-11;
  if(!old_first_rejection && std::abs(d->time-test_step*.002)>1e-10)
    old_first_rejection=test_step;
}
#define clock_gettime virtual_gettime
#define clock_nanosleep virtual_nanosleep
#define mj_step checked_step
#include "PRODUCTION_SOURCE"
#undef mj_step
#undef clock_nanosleep
#undef clock_gettime
int main(int argc, char **argv) {
  char error[1024];
  mjModel *m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
  if(!m){std::cerr<<error;return 2;}
  bool passed=true;
  std::cout<<std::setprecision(17)<<"{\"cases\":[";
  const char *names[]={"long_200_seconds","reset_time","missing_tick","extra_tick","incremental_drift","nonzero_initial_time"};
  for(int scenario=0;scenario<6;++scenario) {
    mjData *d=mj_makeData(m);
    const int controls=scenario==0?10000:200;
    if(scenario==5)d->time=1e-6;
    test_time=0;test_step=0;test_fault=scenario;old_first_rejection=0;
    Shared shared;
    std::vector<double> states((controls*10+1)*60),targets(controls*10*23),torques(controls*10*23);
    std::vector<int64_t> timing(controls*10*7);
    double zero[23]{},one[23],summary[10]{};
    std::fill(one,one+23,1);
    const int steps=clock_run(&shared,m,d,controls,0,zero,zero,one,one,zero,one,zero,
      states.data(),targets.data(),torques.data(),timing.data(),summary);
    const bool good=scenario==0 ? (steps==100000 && summary[3]==0 && old_first_rejection>0)
      : (summary[3]==1 && (scenario==4 ? steps<=7 : steps==(scenario==5?1:100)));
    passed &= good;
    if(scenario)std::cout<<",";
    std::cout<<"{\"name\":\""<<names[scenario]<<"\",\"steps\":"<<steps
      <<",\"mujoco_time\":"<<d->time<<",\"physical_failure\":"<<summary[3]
      <<",\"old_multiplication_first_rejection\":"<<old_first_rejection
      <<",\"passed\":"<<(good?"true":"false")<<"}";
    mj_deleteData(d);
  }
  std::cout<<"],\"passed\":"<<(passed?"true":"false")<<"}\n";
  mj_deleteModel(m);return passed?0:1;
}
'''.replace('PRODUCTION_SOURCE', source.as_posix())
(out / 'clock_harness.cpp').write_text(harness)
bodies = ''.join(f'<body><joint name="j{i}" type="hinge" axis="0 0 1" range="-2 2"/><geom type="sphere" size=".01" mass=".01"/></body>' for i in range(23))
motors = ''.join(f'<motor joint="j{i}"/>' for i in range(23))
(out / 'stationary_clock_test.xml').write_text('<mujoco><compiler angle="radian"/><option timestep=".002" gravity="0 0 0"/><default><geom contype="0" conaffinity="0"/></default><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/>' + bodies + '</body></worldbody><actuator>' + motors + '</actuator></mujoco>')
subprocess.run(['g++', '-O3', '-std=c++17', '-I' + str(package / 'include'),
                str(out / 'clock_harness.cpp'), str(library), '-Wl,-rpath,' + str(package),
                '-o', str(out / 'clock_harness')], check=True)
result = subprocess.run([str(out / 'clock_harness'), str(out / 'stationary_clock_test.xml')],
                        check=True, capture_output=True, text=True)
report = json.loads(result.stdout)
report.update(production_source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              mujoco=mujoco.__version__, tolerance=1e-10, scheduling_interval_ns=2_000_000,
              test_scope='Production clock_run and real MuJoCo stepping with virtual wall-clock waits; no controller behavior or real-time qualification')
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
