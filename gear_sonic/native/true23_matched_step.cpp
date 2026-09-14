// Shared diagnostic plant step. PD and stop conditions from the archived
// native_clock_v1/braking_v5 loop; no controller, filtering or wall clock.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
extern "C" int matched_step(mjModel* m, mjData* d, const double* target,
    const double* kp, const double* kd, const double* effort,
    const double* velocity, double expected_time, double* requested) {
  for (int j=0;j<23;++j) {
    requested[j]=kp[j]*(target[j]-d->qpos[j+7])-kd[j]*d->qvel[j+6];
    d->ctrl[j]=std::clamp(requested[j],-effort[j],effort[j]);
  }
  mj_step(m,d);
  int failure=0;
  for(int j=0;j<30;++j) if(!std::isfinite(d->qpos[j])) failure|=1;
  for(int j=0;j<29;++j) if(!std::isfinite(d->qvel[j])) failure|=1;
  for(int j=0;j<23;++j) {
    if(d->qpos[7+j]<m->jnt_range[2*(j+1)]-1e-6 ||
       d->qpos[7+j]>m->jnt_range[2*(j+1)+1]+1e-6) failure|=2;
    if(std::abs(d->qvel[6+j])/velocity[j]>1) failure|=4;
    if(std::abs(d->qfrc_actuator[6+j])/effort[j]>1+1e-9) failure|=8;
  }
  for(int j=0;j<mjNWARNING;++j) if(d->warning[j].number) failure|=16;
  double norm=0;for(int j=3;j<7;++j)norm+=d->qpos[j]*d->qpos[j];
  if(std::abs(std::sqrt(norm)-1)>1e-10) failure|=32;
  double tilt=std::acos(std::clamp(1-2*(d->qpos[4]*d->qpos[4]+d->qpos[5]*d->qpos[5]),-1.,1.));
  if(d->qpos[2]<.25 || tilt>1.2) failure|=64;
  if(std::abs(d->time-expected_time)>1e-10) failure|=128;
  return failure;
}
