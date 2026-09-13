
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
#include "/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic/native/true23_clock.cpp"
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
