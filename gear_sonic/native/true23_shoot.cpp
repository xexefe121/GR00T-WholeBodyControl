// Short native23 candidate rollouts. Simulation only; no clock or hardware I/O.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>

struct Shoot {
  const mjModel* model;
  mjData* data;
  double kp[23],kd[23],effort[23],velocity[23],limits[46],offsets[9];
  int ids[6]; // root, two feet, three hand/head task bodies
};
static double square(double x){return x*x;}
static bool invalid(Shoot* s) {
  auto* d=s->data;
  for(int i=0;i<30;i++)if(!std::isfinite(d->qpos[i]))return true;
  for(int i=0;i<29;i++)if(!std::isfinite(d->qvel[i]))return true;
  for(int i=0;i<23;i++)if(std::abs(d->qvel[6+i])>s->velocity[i] ||
      d->qpos[7+i]<s->limits[2*i]-1e-6 || d->qpos[7+i]>s->limits[2*i+1]+1e-6)return true;
  for(int i=0;i<mjNWARNING;i++)if(d->warning[i].number)return true;
  return d->qpos[2]<.25 || 1-2*(square(d->qpos[4])+square(d->qpos[5]))<std::cos(1.2);
}
static double cost(Shoot* s,const double* g,double t,const double* initial_com) {
  auto* d=s->data;auto* m=s->model;
  mj_kinematics(m,d);mj_comPos(m,d);
  double goal_root[3],root_cost=0,velocity_cost=0;
  for(int j=0;j<3;j++) {
    goal_root[j]=g[46+j]+t*g[49+j];
    root_cost+=square(d->qpos[j]-goal_root[j]);
    velocity_cost+=square(d->qvel[j]-g[49+j]);
  }
  double rot[9],delta[4],goal_quat[4],axis[3]={g[56],g[57],g[58]};
  double angle=mju_normalize3(axis)*t;
  mju_axisAngle2Quat(delta,axis,angle);mju_mulQuat(goal_quat,delta,g+52);mju_quat2Mat(rot,goal_quat);
  double orientation=0;
  for(int j=0;j<9;j++)orientation+=square(d->xmat[9*s->ids[0]+j]-rot[j]);
  double legs=0,all_joints=0;
  for(int j=0;j<23;j++) {
    double e=square(d->qpos[7+j]-(g[j]+t*g[23+j]));
    all_joints+=e;if(j<12)legs+=e;
  }
  double feet=0,tasks=0;
  for(int k=0;k<2;k++)for(int j=0;j<3;j++)
    feet+=square((d->xpos[3*s->ids[1+k]+j]-d->qpos[j])-(g[59+3*k+j]+t*g[65+3*k+j]-goal_root[j]));
  for(int k=0;k<3;k++) {
    double point[3];mju_mulMatVec3(point,d->xmat+9*s->ids[3+k],s->offsets+3*k);
    for(int j=0;j<3;j++) {
      point[j]+=d->xpos[3*s->ids[3+k]+j];
      tasks+=square((point[j]-d->qpos[j])-(g[71+3*k+j]+t*g[80+3*k+j]-goal_root[j]))/(k==2?.01:.0225);
    }
  }
  // Penalize predicted capture point outside the actual ground support box.
  // This is an optimization cost, never a replacement for physical checks.
  double low[2]={1e9,1e9},high[2]={-1e9,-1e9};bool support=false;
  for(int k=0;k<d->ncon;k++) {
    const auto& c=d->contact[k];if(c.dist>.003)continue;
    int b0=m->geom_bodyid[c.geom[0]],b1=m->geom_bodyid[c.geom[1]];
    bool foot=(b0==s->ids[1]||b0==s->ids[2]||b1==s->ids[1]||b1==s->ids[2]);
    if(!foot || (b0!=0 && b1!=0))continue;
    support=true;
    for(int j=0;j<2;j++){low[j]=std::min(low[j],c.pos[j]-.025);high[j]=std::max(high[j],c.pos[j]+.025);}
  }
  double capture=0;
  const double* com=d->subtree_com+3*s->ids[0];
  const double omega=std::sqrt(9.81/std::max(.3,com[2]));
  if(support)for(int j=0;j<2;j++) {
    double cp=com[j]+(com[j]-initial_com[j])/std::max(.002,t)/omega;
    capture+=square(cp-std::clamp(cp,low[j],high[j]))/.01;
  }
  else capture=10;
  return 2*root_cost/.04+orientation/.137+.5*velocity_cost/.09+
      feet/.0144+2*tasks/3+3*legs/(12*.0225)+all_joints/(23*.04)+2*capture;
}
extern "C" {
void* shoot_create(const mjModel* model,const double* kp,const double* kd,const double* effort,
    const double* velocity,const double* limits,const int* ids,const double* offsets) {
  if(!model || model->nq!=30 || model->nv!=29 || model->nu!=23)return nullptr;
  auto* s=new(std::nothrow) Shoot();if(!s)return nullptr;
  s->model=model;s->data=mj_makeData(model);if(!s->data){delete s;return nullptr;}
  std::copy(kp,kp+23,s->kp);std::copy(kd,kd+23,s->kd);std::copy(effort,effort+23,s->effort);
  std::copy(velocity,velocity+23,s->velocity);std::copy(limits,limits+46,s->limits);
  std::copy(ids,ids+6,s->ids);std::copy(offsets,offsets+9,s->offsets);return s;
}
void shoot_destroy(void* context){auto* s=static_cast<Shoot*>(context);if(s){mj_deleteData(s->data);delete s;}}
int shoot_evaluate(void* context,const double* q,const double* v,const double* targets,int count,int steps,
    const double* goal,double* scores,double* terminals) {
  auto* s=static_cast<Shoot*>(context);if(!s||count<1||count>64||steps<10||steps>100)return -1;
  auto* d=s->data;auto* m=s->model;int selected=-1;double best=1e100;
  for(int n=0;n<count;n++) {
    mj_resetData(m,d);std::copy(q,q+30,d->qpos);std::copy(v,v+29,d->qvel);mj_forward(m,d);
    double initial_com[3];std::copy(d->subtree_com+3*s->ids[0],d->subtree_com+3*s->ids[0]+3,initial_com);
    double sum=0;int samples=0;bool failed=invalid(s);
    for(int step=0;step<steps&&!failed;step++) {
      for(int j=0;j<23;j++) {
        double target=std::clamp(targets[n*23+j],s->limits[2*j]+.06,s->limits[2*j+1]-.06);
        double band=std::min(.1,.2*(s->limits[2*j+1]-s->limits[2*j]));
        double penetration=d->qpos[7+j]-std::clamp(d->qpos[7+j],s->limits[2*j]+band,s->limits[2*j+1]-band);
        double outward=penetration*d->qvel[6+j]>0?d->qvel[6+j]:0;
        d->ctrl[j]=std::clamp(s->kp[j]*(target-d->qpos[7+j])-s->kd[j]*d->qvel[6+j]-100*penetration-2*outward,-s->effort[j],s->effort[j]);
      }
      mj_step(m,d);failed=invalid(s);
      if(!failed&&((step+1)%10==0||step==steps-1)){sum+=cost(s,goal,(step+1)*m->opt.timestep,initial_com);samples++;}
    }
    double movement=0;for(int j=0;j<23;j++)movement+=square((targets[n*23+j]-targets[j])/.25);
    scores[n]=failed?1e9:sum/std::max(1,samples)+.02*movement/23;
    std::copy(d->qpos,d->qpos+30,terminals+n*59);std::copy(d->qvel,d->qvel+29,terminals+n*59+30);
    if(scores[n]<best){selected=n;best=scores[n];}
  }
  return selected;
}
}
