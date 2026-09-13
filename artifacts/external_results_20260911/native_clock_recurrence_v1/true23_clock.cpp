// Independent native 500 Hz simulation plant. No filesystem I/O in the loop.
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

struct Observation {
  std::atomic_flag lock = ATOMIC_FLAG_INIT;
  int id = -1, fault = 0;
  int64_t published = 0;
  double values[382]{}; // qpos30, qvel29, previous23, history300
};
struct Result {
  std::atomic_flag lock = ATOMIC_FLAG_INIT;
  int id = -1;
  int64_t start = 0, finish = 0;
  double target[23]{};
};
struct Shared {
  Observation observation;
  Result active, standing;
  std::atomic<int> stop{0}, fault{0};
};
static int64_t now_ns() { timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return int64_t(t.tv_sec)*1000000000+t.tv_nsec; }
static int64_t spin_ns=0;
static void wait_ns(int64_t deadline) {
  int64_t sleep_deadline=deadline-spin_ns;
  timespec t{sleep_deadline/1000000000,sleep_deadline%1000000000};
  while (clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&t,nullptr)==EINTR) {}
  while(now_ns()<deadline)std::atomic_signal_fence(std::memory_order_seq_cst);
}
extern "C" {
void clock_set_spin_ns(int64_t value) { spin_ns=std::clamp(value,int64_t(0),int64_t(500000)); }
size_t clock_shared_size() { return sizeof(Shared); }
void clock_initialize(void *p) { new(p) Shared(); }
void clock_stop(void *p) { static_cast<Shared*>(p)->stop.store(1); }
void clock_latch_fault(void *p) { static_cast<Shared*>(p)->fault.store(1); }
int clock_read_observation(void *p,int *id,int *fault,int64_t *published,double *values) {
  auto &x=static_cast<Shared*>(p)->observation;
  if(x.lock.test_and_set(std::memory_order_acquire))return 0;
  *id=x.id;*fault=x.fault;*published=x.published;std::copy(x.values,x.values+382,values);
  x.lock.clear(std::memory_order_release);return 1;
}
int clock_publish_observation(void *p,int id,int fault,int64_t published,const double *values) {
  auto &x=static_cast<Shared*>(p)->observation;
  bool acquired=false;for(int attempt=0;attempt<64;++attempt)if(!x.lock.test_and_set(std::memory_order_acquire)){acquired=true;break;}
  if(!acquired)return 0;
  x.id=id;x.fault=fault;x.published=published;std::copy(values,values+382,x.values);
  x.lock.clear(std::memory_order_release);return 1;
}
int clock_publish_result(void *p,int standing,int id,int64_t start,int64_t finish,const double *target) {
  auto *s=static_cast<Shared*>(p);auto &x=standing?s->standing:s->active;
  if(x.lock.test_and_set(std::memory_order_acquire))return 0;
  x.id=id;x.start=start;std::copy(target,target+23,x.target);x.finish=now_ns();
  x.lock.clear(std::memory_order_release);return 1;
}
int clock_read_result(void *p,int standing,int *id,int64_t *start,int64_t *finish,double *target) {
  auto *s=static_cast<Shared*>(p);auto &x=standing?s->standing:s->active;
  if(x.lock.test_and_set(std::memory_order_acquire))return 0;
  *id=x.id;*start=x.start;*finish=x.finish;std::copy(x.target,x.target+23,target);
  x.lock.clear(std::memory_order_release);return 1;
}
// Returns executed steps. All output arrays are allocated by the caller.
int clock_run(void *memory,mjModel *m,mjData *d,int controls,int64_t epoch,
              const double *kp,const double *kd,const double *effort,const double *velocity,
              const double *defaults,const double *training_effort,const double *initial_target,
              double *states,double *targets,double *torques,int64_t *timing,double *summary) {
  auto *shared=static_cast<Shared*>(memory);
  double current[23],prior[23]{},history[300]{},begin[59],incoming[23]{},observed[382]{};
  std::copy(initial_target,initial_target+23,current);
  std::copy(d->qpos,d->qpos+30,states);std::copy(d->qvel,d->qvel+29,states+30);states[59]=d->time;
  std::copy(states,states+59,begin);
  int command_id=0,steps=0,misses=0,publications_missed=0,physical_failure=0,plant_misses=0;
  double peak_speed=0,peak_range=0,peak_effort=0;
  // MuJoCo advances time by repeated addition. Keep an independent recurrence
  // from zero; multiplying the step count falsely fails long runs from rounding.
  double expected_time=0;
  int64_t max_late=0,first_fault=-1;
  for(int step=0;step<controls*10 && !shared->stop.load();++step) {
    const int control=step/10;
    const int64_t deadline=epoch+int64_t(step+1)*2000000;
    wait_ns(deadline);int64_t wake=now_ns();max_late=std::max(max_late,wake-deadline);
    if(wake>deadline+2000000)++plant_misses;
    int balance=shared->fault.load() || control<250 || control>=1269;
    if(shared->fault.load() && first_fault<0)first_fault=control;
    int result_id=-1;int64_t started=0,finished=0;double proposed[23];
    if(clock_read_result(memory,balance,&result_id,&started,&finished,proposed) && result_id==control) {
      bool finite=true;
      for(int j=0;j<23;++j)finite &= std::isfinite(proposed[j]) && proposed[j]>=m->jnt_range[2*(j+1)]-1e-12 && proposed[j]<=m->jnt_range[2*(j+1)+1]+1e-12;
      if(finite && finished<=epoch+int64_t(control+1)*20000000) {
        std::copy(proposed,proposed+23,current);command_id=result_id;
      } else if(!shared->fault.exchange(1)) first_fault=control;
    }
    if(step%10==9 && command_id!=control) {
      ++misses;
      if(!shared->fault.exchange(1))first_fault=control;
    }
    int64_t native_start=now_ns();
    for(int j=0;j<23;++j) {
      d->ctrl[j]=std::clamp(kp[j]*(current[j]-d->qpos[j+7])-kd[j]*d->qvel[j+6],-effort[j],effort[j]);
      targets[step*23+j]=current[j];torques[step*23+j]=d->ctrl[j];
    }
    mj_step(m,d);int64_t finish=now_ns();
    expected_time+=.002;
    timing[step*7]=deadline;timing[step*7+1]=wake;timing[step*7+2]=native_start;
    timing[step*7+3]=finish;timing[step*7+4]=command_id;timing[step*7+5]=control;timing[step*7+6]=shared->fault.load();
    auto *state=states+(step+1)*60;std::copy(d->qpos,d->qpos+30,state);std::copy(d->qvel,d->qvel+29,state+30);state[59]=d->time;
    for(int j=0;j<23;++j) {
      peak_speed=std::max(peak_speed,std::abs(d->qvel[j+6])/velocity[j]);
      peak_range=std::max(peak_range,std::max(m->jnt_range[2*(j+1)]-d->qpos[j+7],d->qpos[j+7]-m->jnt_range[2*(j+1)+1]));
      peak_effort=std::max(peak_effort,std::abs(d->qfrc_actuator[j+6])/effort[j]);
    }
    double norm=0;for(int j=3;j<7;++j)norm+=d->qpos[j]*d->qpos[j];
    const double tilt=std::acos(std::clamp(1-2*(d->qpos[4]*d->qpos[4]+d->qpos[5]*d->qpos[5]),-1.,1.));
    bool finite=true;for(int j=0;j<59;++j)finite &= std::isfinite(state[j]);
    bool warning=false;for(int j=0;j<mjNWARNING;++j)warning |= d->warning[j].number!=0;
    physical_failure= !finite || warning || peak_range>1e-6 || peak_speed>1 || peak_effort>1+1e-9 ||
        d->qpos[2]<.25 || tilt>1.2 || std::abs(std::sqrt(norm)-1)>1e-10 || std::abs(d->time-expected_time)>1e-10;
    steps=step+1;
    if(physical_failure || wake-deadline>200000000)break;
    if(step%10==9 && control+1<controls) {
      // Update history once from previous control-boundary measured state and
      // the actual command that reached the plant; no proposed-action history.
      double terms[75];std::copy(incoming,incoming+23,terms);
      for(int j=0;j<3;++j)terms[23+j]=float(begin[33+j]*.25);
      for(int j=0;j<23;++j){terms[26+j]=float(begin[7+j]-defaults[j]);terms[49+j]=float(begin[36+j]);}
      const double w=begin[3],x=begin[4],y=begin[5],z=begin[6];
      terms[72]=float(-2*(x*z-w*y));terms[73]=float(-2*(y*z+w*x));terms[74]=float(-1+2*(x*x+y*y));
      int offset=0,term=0;for(int width:{23,3,23,23,3}) {
        for(int row=3;row>0;--row)std::copy(history+offset+(row-1)*width,history+offset+row*width,history+offset+row*width);
        std::copy(terms+term,terms+term+width,history+offset);offset+=4*width;term+=width;
      }
      for(int j=0;j<23;++j)prior[j]=float((current[j]-defaults[j])*kp[j]/(.25*training_effort[j]));
      std::copy(state,state+59,observed);std::copy(prior,prior+23,observed+59);std::copy(history,history+300,observed+82);
      if(!clock_publish_observation(memory,control+1,shared->fault.load(),now_ns(),observed))++publications_missed;
      std::copy(state,state+59,begin);std::copy(prior,prior+23,incoming);
    }
  }
  summary[0]=steps;summary[1]=misses;summary[2]=plant_misses;summary[3]=physical_failure;
  summary[4]=first_fault;summary[5]=peak_speed;summary[6]=peak_range;summary[7]=peak_effort;
  summary[8]=max_late*1e-6;summary[9]=publications_missed;
  shared->stop.store(1);return steps;
}
}
