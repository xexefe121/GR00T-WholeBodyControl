/* Offline sampled continuation, never a robot publisher or safety proof. */
#include <mujoco/mujoco.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  const mjModel* m;
  mjData *seed, *d;
  int j;
  double kp[23],kd[23],effort[23],velocity[23],limits[46];
} Predictor;

void* braking_create(const mjModel* m,int j,const double* kp,const double* kd,
                     const double* effort,const double* velocity,const double* limits) {
  if(m->nq!=30 || m->nv!=29 || m->nu!=23 || j<0 || j>=23 || fabs(m->opt.timestep-.002)>1e-12) return NULL;
  Predictor* p=calloc(1,sizeof(Predictor)); p->m=m;p->j=j;
  p->seed=mj_makeData(m);p->d=mj_makeData(m);
  memcpy(p->kp,kp,23*sizeof(double));memcpy(p->kd,kd,23*sizeof(double));
  memcpy(p->effort,effort,23*sizeof(double));memcpy(p->velocity,velocity,23*sizeof(double));
  memcpy(p->limits,limits,46*sizeof(double));return p;
}
void braking_destroy(Predictor* p) {if(p){mj_deleteData(p->seed);mj_deleteData(p->d);free(p);}}
static double clip(double x,double lo,double hi){return fmax(lo,fmin(hi,x));}
static void reset(Predictor* p,const double* q,const double* v) {
  mj_resetData(p->m,p->seed);memcpy(p->seed->qpos,q,30*sizeof(double));
  memcpy(p->seed->qvel,v,29*sizeof(double));mj_forward(p->m,p->seed);
}
/* Fixed backup: critically damped acceleration toward an interior capture goal,
   effective free-coordinate inertia and current model bias compensation.
   This modifies the target, not the physical PD gains or effort caps. */
static double backup(Predictor* p,mjData* d,double goal) {
  int k=6+p->j;double unit[29]={0},inverse[29];unit[k]=1;
  mj_forward(p->m,d);mj_solveM(p->m,d,inverse,unit,1);
  double inertia=1/fmax(inverse[k],1e-9);
  double accel=400*(goal-d->qpos[7+p->j])-40*d->qvel[k];
  double torque=clip(inertia*accel+d->qfrc_bias[k]-d->qfrc_passive[k],-p->effort[p->j],p->effort[p->j]);
  return clip(d->qpos[7+p->j]+(torque+p->kd[p->j]*d->qvel[k])/p->kp[p->j],p->limits[2*p->j],p->limits[2*p->j+1]);
}
double braking_target(Predictor* p,const double* q,const double* v,double goal) {
  reset(p,q,v);return backup(p,p->seed,goal);
}
/* Columns: pass, failstep, reason bitmask, min position margin, max speed ratio,
   max applied effort ratio, max raw effort ratio, terminal q, terminal v,
   terminal achieved, limiting joint. Reasons 1 position/reserve,2 speed,4 effort,
   8 fall,16 nonfinite,32 engine,64 quaternion/clock,128 terminal backup missing. */
void braking_evaluate(Predictor* p,const double* q,const double* v,const double* previous,
                      const double* first,const double* nominal,double goal,double* rows) {
  reset(p,q,v);
  for(int delay0=0;delay0<7;delay0++)for(int delay1=0;delay1<7;delay1++) {
    mjData* d=p->d;mj_copyData(d,p->m,p->seed);
    double* row=rows+(delay0*7+delay1)*11;memset(row,0,11*sizeof(double));row[3]=1e9;row[10]=-1;
    double last[23],target[23];memcpy(last,previous,23*sizeof(double));memcpy(target,first,23*sizeof(double));
    int terminal_streak=0;
    for(int step=0;step<40;step++) {
      int cycle=step/10,substep=step%10,delay=cycle?delay1:delay0;
      if(step && substep==0) {
        memcpy(last,target,23*sizeof(double));memcpy(target,nominal,23*sizeof(double));
        target[p->j]=backup(p,d,goal);
      }
      const double* applied=substep<delay?last:target;
      for(int j=0;j<23;j++) {
        double raw=p->kp[j]*(applied[j]-d->qpos[7+j])-p->kd[j]*d->qvel[6+j];
        row[6]=fmax(row[6],fabs(raw)/p->effort[j]);d->ctrl[j]=clip(raw,-p->effort[j],p->effort[j]);
      }
      mj_step(p->m,d);int reason=0;
      for(int j=0;j<30;j++)if(!isfinite(d->qpos[j]))reason|=16;
      for(int j=0;j<29;j++)if(!isfinite(d->qvel[j])||!isfinite(d->qfrc_actuator[j]))reason|=16;
      for(int j=0;j<23;j++) {
        double margin=fmin(d->qpos[7+j]-p->limits[2*j],p->limits[2*j+1]-d->qpos[7+j]);
        if(margin<row[3]){row[3]=margin;row[10]=j;}
        row[4]=fmax(row[4],fabs(d->qvel[6+j])/p->velocity[j]);
        row[5]=fmax(row[5],fabs(d->qfrc_actuator[6+j])/p->effort[j]);
        if(margin<.0001)reason|=1;
      }
      if(row[4]>1)reason|=2;if(row[5]>1+1e-9)reason|=4;
      double tilt=acos(clip(1-2*(d->qpos[4]*d->qpos[4]+d->qpos[5]*d->qpos[5]),-1,1));
      if(d->qpos[2]<.25||tilt>1.2)reason|=8;
      for(int j=0;j<mjNWARNING;j++)if(d->warning[j].number)reason|=32;
      double norm=0;for(int j=3;j<7;j++)norm+=d->qpos[j]*d->qpos[j];
      if(fabs(sqrt(norm)-1)>1e-10||fabs(d->time-(step+1)*.002)>1e-10)reason|=64;
      if(reason){row[1]=step+1;row[2]=reason;break;}
      if(substep==9 && cycle>=2) {
        double margin=fmin(d->qpos[7+p->j]-p->limits[2*p->j],p->limits[2*p->j+1]-d->qpos[7+p->j]);
        terminal_streak=(margin>=.02 && fabs(d->qvel[6+p->j])<=.25)?terminal_streak+1:0;
      }
    }
    row[7]=d->qpos[7+p->j];row[8]=d->qvel[6+p->j];row[9]=terminal_streak>=2;
    if(!row[2]&&!row[9]){row[1]=40;row[2]=128;}
    row[0]=row[2]==0;
  }
}
