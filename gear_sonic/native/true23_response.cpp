// Independent native MuJoCo predictions for target-response feedback. Sim only.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <new>

struct Response {
  const mjModel* model;
  mjData* data[64]{};
  double kp[23],kd[23],effort[23],velocity[23],limits[46],offsets[15];
  int ids[5];
};
static bool invalid(Response* s,mjData* d) {
  for(int j=0;j<30;j++)if(!std::isfinite(d->qpos[j]))return true;
  for(int j=0;j<29;j++)if(!std::isfinite(d->qvel[j]))return true;
  for(int j=0;j<23;j++)if(std::abs(d->qvel[6+j])>s->velocity[j] ||
    d->qpos[7+j]<s->limits[2*j]-1e-6 || d->qpos[7+j]>s->limits[2*j+1]+1e-6)return true;
  for(int j=0;j<mjNWARNING;j++)if(d->warning[j].number)return true;
  return d->qpos[2]<.25 || 1-2*(d->qpos[4]*d->qpos[4]+d->qpos[5]*d->qpos[5])<std::cos(1.2);
}
extern "C" {
void response_destroy(void* context) {
  auto* s=static_cast<Response*>(context);
  if(s){for(auto* d:s->data)if(d)mj_deleteData(d);delete s;}
}
void* response_create(const mjModel* model,const double* kp,const double* kd,const double* effort,
  const double* velocity,const double* limits,const int* ids,const double* offsets) {
  if(!model || model->nq!=30 || model->nv!=29 || model->nu!=23)return nullptr;
  auto* s=new(std::nothrow) Response();if(!s)return nullptr;s->model=model;
  for(auto*& d:s->data){d=mj_makeData(model);if(!d){response_destroy(s);return nullptr;}}
  std::copy(kp,kp+23,s->kp);std::copy(kd,kd+23,s->kd);std::copy(effort,effort+23,s->effort);
  std::copy(velocity,velocity+23,s->velocity);std::copy(limits,limits+46,s->limits);
  std::copy(ids,ids+5,s->ids);std::copy(offsets,offsets+15,s->offsets);return s;
}
int response_predict(void* context,const double* q,const double* v,const double* targets,
  int count,int steps,double* outputs,int* failures) {
  auto* s=static_cast<Response*>(context);if(!s||count<1||count>64||steps<1||steps>100)return -1;
  auto* m=s->model;
  #pragma omp parallel for num_threads(4) schedule(static)
  for(int n=0;n<count;n++) {
    auto* d=s->data[n];mj_resetData(m,d);
    std::copy(q,q+30,d->qpos);std::copy(v,v+29,d->qvel);mj_forward(m,d);
    bool failed=invalid(s,d);
    for(int step=0;step<steps&&!failed;step++) {
      for(int j=0;j<23;j++) {
        double target=std::clamp(targets[n*23+j],s->limits[2*j]+.06,s->limits[2*j+1]-.06);
        double band=std::min(.1,.2*(s->limits[2*j+1]-s->limits[2*j]));
        double penetration=d->qpos[7+j]-std::clamp(d->qpos[7+j],s->limits[2*j]+band,s->limits[2*j+1]-band);
        double outward=penetration*d->qvel[6+j]>0?d->qvel[6+j]:0;
        d->ctrl[j]=std::clamp(s->kp[j]*(target-d->qpos[7+j])-s->kd[j]*d->qvel[6+j]-100*penetration-2*outward,
          -s->effort[j],s->effort[j]);
      }
      mj_step(m,d);failed=invalid(s,d);
    }
    failures[n]=failed?1:0;
    double* out=outputs+n*89;
    std::copy(d->qpos,d->qpos+30,out);std::copy(d->qvel,d->qvel+29,out+30);
    mj_kinematics(m,d);mj_comPos(m,d);
    for(int k=0;k<5;k++) {
      double point[3],jac[87];mju_mulMatVec3(point,d->xmat+9*s->ids[k],s->offsets+3*k);
      for(int j=0;j<3;j++)point[j]+=d->xpos[3*s->ids[k]+j];
      mj_jac(m,d,jac,nullptr,point,s->ids[k]);
      std::copy(point,point+3,out+59+3*k);mju_mulMatVec(out+74+3*k,jac,d->qvel,3,29);
    }
  }
  return 0;
}
}
