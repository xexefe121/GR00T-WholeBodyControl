// Received-only candidate prediction with native factory feedback retained.
// Each candidate owns its mjData; all share an immutable native23 model.
#include "true23_shoot.cpp"
#include <vector>
#include <omp.h>

struct FeedbackShoot {
  Shoot* worlds[32]{};
  std::vector<float> weights;
  double defaults[23],scale[12];
  std::vector<float> learned_weights;
  float goal_mean[1020]{},goal_scale[1020]{};
  double reference_default[23]{};
  std::vector<double> received_goals; // stage, eight received/predicted slots,130 fields
  int received_stages=0;
  double debug_trace[30*82]{};
  int debug_controls=0;
};
static void layer(const float*& weights,const float* input,float* output,int ni,int no,int activation) {
  const float* bias=weights+ni*no;
  for(int o=0;o<no;o++) {
    float sum=0;const float* row=weights+o*ni;
    #pragma omp simd reduction(+:sum)
    for(int i=0;i<ni;i++)sum+=row[i]*input[i];
    sum+=bias[o];
    output[o]=activation==1?(sum>=0?sum:std::expm1(sum)):(activation==2?std::tanh(sum):sum);
  }
  weights=bias+no;
}
static void factory(const FeedbackShoot* f,const float* history,const float* command,float* output) {
  const float* w=f->weights.data();float a[256],b[256],joined[72];
  layer(w,history,a,210,256,1);layer(w,a,b,256,128,1);layer(w,b,joined,128,64,2);
  std::copy(command,command+8,joined+64);
  layer(w,joined,a,72,256,1);layer(w,a,b,256,128,1);layer(w,b,a,128,32,1);layer(w,a,output,32,12,0);
}
static void received_features(const FeedbackShoot* f,const double* q,const double* v,const double* refs,float* out) {
  double rot[9];mju_quat2Mat(rot,q+3);double yaw=std::atan2(rot[3],rot[0]),c=std::cos(yaw),s=std::sin(yaw);
  int n=0;
  for(int j=0;j<23;j++)out[n++]=q[7+j]-f->reference_default[j];
  for(int j=0;j<23;j++)out[n++]=v[6+j];
  for(int j=0;j<3;j++)out[n++]=v[3+j];
  for(int j=0;j<3;j++)out[n++]=-rot[6+j];
  auto vector=[&](const double* x,const double* origin=nullptr) {
    double a=x[0]-(origin?origin[0]:0),b=x[1]-(origin?origin[1]:0),z=x[2]-(origin?origin[2]:0);
    out[n++]=c*a+s*b;out[n++]=-s*a+c*b;out[n++]=z;
  };
  auto rotation=[&](const double* r) {
    for(int i=0;i<3;i++)for(int j=0;j<2;j++)
      out[n++]=i==0?c*r[j]+s*r[3+j]:i==1?-s*r[j]+c*r[3+j]:r[6+j];
  };
  vector(v);out[n++]=q[2];
  for(int k=0;k<8;k++) {
    const double* r=refs+130*k;
    for(int j=0;j<46;j++)out[n++]=r[j];
    vector(r+46,q);rotation(r+49);vector(r+58);vector(r+61);
    for(int j=0;j<2;j++)vector(r+64+3*j,q);
    for(int j=0;j<2;j++)vector(r+70+3*j);
    for(int j=0;j<3;j++)vector(r+76+3*j,q);
    for(int j=0;j<3;j++)rotation(r+85+9*j);
    for(int j=0;j<3;j++)vector(r+112+3*j);
    for(int j=0;j<3;j++)vector(r+121+3*j);
  }
}
static void learned(const FeedbackShoot* f,const Shoot* s,const float* features,const float* command,const float* raw,float* output) {
  float head[1020],a[256],b[128],delta[12];
  std::copy(features,features+1000,head);std::copy(command,command+8,head+1000);std::copy(raw,raw+12,head+1008);
  for(int j=0;j<1020;j++)head[j]=std::clamp((head[j]-f->goal_mean[j])/f->goal_scale[j],-20.f,20.f);
  const float* w=f->learned_weights.data();layer(w,head,a,1020,256,1);layer(w,a,b,256,128,1);layer(w,b,delta,128,12,0);
  for(int j=0;j<23;j++) {
    float desired;
    if(j<12)desired=float(f->defaults[j])+float(f->scale[j])*raw[j]+delta[j]*(float(s->limits[2*j+1])-float(s->limits[2*j]))*.5f;
    else desired=features[56+j]+float(s->kd[j]/s->kp[j])*features[79+j];
    desired=std::clamp(desired,float(s->limits[2*j])+.06f,float(s->limits[2*j+1])-.06f);
    output[j]=desired-float(f->defaults[j]);
  }
}
static void future_target(FeedbackShoot* f,Shoot* s,float* history,double* velocity,double& phase,bool& walking,
    const double* previous,const double* residual,const double* candidate,const double* goal,double t,double* target) {
  auto* d=s->data;double rot[9];mju_quat2Mat(rot,d->qpos+3);
  double yaw=std::atan2(rot[3],rot[0]),cy=std::cos(yaw),sy=std::sin(yaw);
  double root[2]={goal[46]+t*goal[49],goal[47]+t*goal[50]};
  double xy[2]={goal[49]+2.5*(root[0]-d->qpos[0]),goal[50]+2.5*(root[1]-d->qpos[1])};
  double gr[9];mju_quat2Mat(gr,goal+52);double gy=std::atan2(gr[3],gr[0])+t*goal[58];
  const double* received=nullptr;
  if(!f->learned_weights.empty()) {
    received=f->received_goals.data()+std::lround(t/.02)*8*130;
    for(int j=0;j<2;j++)xy[j]=received[58+j]+2.5*(received[46+j]-d->qpos[j]);
    gy=std::atan2(received[52],received[49]);
  }
  double dy=std::atan2(std::sin(gy-yaw),std::cos(gy-yaw));
  double wanted[3]={cy*xy[0]+sy*xy[1],-sy*xy[0]+cy*xy[1],goal[58]+3*dy};
  if(received)wanted[2]=received[63]+3*dy;
  const double cap[3]={1.2,1.,6.},rate[3]={.24,.24,.8};
  for(int j=0;j<3;j++)velocity[j]+=std::clamp(std::clamp(wanted[j],-cap[j],cap[j])-velocity[j],-rate[j],rate[j]);
  double magnitude=std::max(std::hypot(velocity[0],velocity[1]),std::abs(velocity[2]));
  walking=walking?magnitude>=.06:magnitude>.12;double frequency=walking?1.2:0;
  phase=walking?std::fmod(phase+frequency*.02,1.):0.;float command[8];
  for(int j=0;j<2;j++) {
    double p=walking?std::fmod(phase+j*.5,1.):0.;
    command[j]=std::sin(2*M_PI*p);command[2+j]=std::cos(2*M_PI*p);
  }
  command[4]=(frequency-1.2)*.5;for(int j=0;j<3;j++)command[5+j]=velocity[j]*.2;
  std::memmove(history,history+42,168*sizeof(float));float* obs=history+168;
  for(int j=0;j<3;j++){obs[j]=d->qvel[3+j]*.2;obs[3+j]=-rot[6+j];}
  for(int j=0;j<12;j++){obs[6+j]=d->qpos[7+j]-f->defaults[j];obs[18+j]=d->qvel[6+j]*.05;obs[30+j]=previous[j]-f->defaults[j];}
  for(int j=0;j<42;j++)obs[j]=std::clamp(obs[j],-10.f,10.f);
  float raw[12];factory(f,history,command,raw);
  if(received) {
    float features[1000],action[23];received_features(f,d->qpos,d->qvel,received,features);
    learned(f,s,features,command,raw,action);
    for(int j=0;j<23;j++) {
      float decoded=float(f->defaults[j])+action[j];
      double desired=std::clamp(double(decoded),s->limits[2*j]+.06,s->limits[2*j+1]-.06);
      if(j<12)desired=previous[j]+.9*(desired-previous[j])+residual[j];
      target[j]=std::clamp(desired,s->limits[2*j]+.06,s->limits[2*j+1]-.06);
    }
    return;
  }
  for(int j=0;j<23;j++) {
    double desired=j<12?f->defaults[j]+f->scale[j]*std::clamp(raw[j],-10.f,10.f)+residual[j]:candidate[j]+t*goal[23+j];
    target[j]=std::clamp(desired,s->limits[2*j]+.06,s->limits[2*j+1]-.06);
  }
}
extern "C" {
void* feedback_create(const mjModel* m,const double* kp,const double* kd,const double* effort,
    const double* velocity,const double* limits,const int* ids,const double* offsets,
    const float* weights,int weight_count,const double* defaults,const double* scale) {
  if(weight_count!=151276)return nullptr;
  auto* f=new(std::nothrow) FeedbackShoot();if(!f)return nullptr;
  for(int i=0;i<32;i++) {
    f->worlds[i]=static_cast<Shoot*>(shoot_create(m,kp,kd,effort,velocity,limits,ids,offsets));
    if(!f->worlds[i]){for(int j=0;j<i;j++)shoot_destroy(f->worlds[j]);delete f;return nullptr;}
  }
  f->weights.assign(weights,weights+weight_count);std::copy(defaults,defaults+23,f->defaults);std::copy(scale,scale+12,f->scale);
  return f;
}
void feedback_destroy(void* context) {
  auto* f=static_cast<FeedbackShoot*>(context);if(f){for(auto* s:f->worlds)shoot_destroy(s);delete f;}
}
void feedback_raw(void* context,const float* history,const float* command,float* output) {
  factory(static_cast<FeedbackShoot*>(context),history,command,output);
}
int feedback_set_learned(void* context,const float* weights,int count,const float* mean,const float* scale,const double* defaults) {
  auto* f=static_cast<FeedbackShoot*>(context);if(!f||count!=295820)return -1;
  for(int j=0;j<1020;j++)if(!std::isfinite(mean[j])||!std::isfinite(scale[j])||scale[j]<=0)return -1;
  f->learned_weights.assign(weights,weights+count);std::copy(mean,mean+1020,f->goal_mean);std::copy(scale,scale+1020,f->goal_scale);
  std::copy(defaults,defaults+23,f->reference_default);return 0;
}
int feedback_set_received_goals(void* context,const double* goals,int stages) {
  auto* f=static_cast<FeedbackShoot*>(context);if(!f||stages<1||stages>31)return -1;
  f->received_goals.assign(goals,goals+stages*8*130);f->received_stages=stages;return 0;
}
int feedback_learned_raw(void* context,const float* history,const float* command,const float* features,float* output) {
  auto* f=static_cast<FeedbackShoot*>(context);if(!f||f->learned_weights.empty())return -1;
  float raw[12];factory(f,history,command,raw);learned(f,f->worlds[0],features,command,raw,output);return 0;
}
int feedback_received_features(void* context,const double* q,const double* v,const double* goals,float* output) {
  auto* f=static_cast<FeedbackShoot*>(context);if(!f)return -1;
  received_features(f,q,v,goals,output);return 0;
}
int feedback_copy_trace(void* context,double* output) {
  auto* f=static_cast<FeedbackShoot*>(context);if(!f)return -1;
  std::copy(f->debug_trace,f->debug_trace+82*f->debug_controls,output);return f->debug_controls;
}
int feedback_evaluate(void* context,const double* q,const double* v,const double* candidates,int count,int steps,
    const double* goal,const float* current_history,const float* current_command,const double* gait,
    double* scores,double* terminals) {
  auto* f=static_cast<FeedbackShoot*>(context);if(!f||count<1||count>32||steps<10||steps>300)return -1;
  if(!f->learned_weights.empty() && f->received_stages<steps/10+1)return -1;
  f->debug_controls=0;
  float initial_raw[12];factory(f,current_history,current_command,initial_raw);
  #pragma omp parallel for num_threads(8) schedule(static)
  for(int n=0;n<count;n++) {
    Shoot* s=f->worlds[n];auto* d=s->data;auto* m=s->model;const double* candidate=candidates+23*n;
    mj_resetData(m,d);std::copy(q,q+30,d->qpos);std::copy(v,v+29,d->qvel);mj_forward(m,d);
    double initial_com[3];std::copy(d->subtree_com+3*s->ids[0],d->subtree_com+3*s->ids[0]+3,initial_com);
    double target[23],previous[23],residual[12],velocity[3]={gait[2],gait[3],gait[4]},phase=gait[0];bool walking=gait[1]!=0;
    float history[210];std::copy(current_history,current_history+210,history);std::copy(candidate,candidate+23,target);
    for(int j=0;j<12;j++)residual[j]=f->learned_weights.empty()?
      candidate[j]-(f->defaults[j]+f->scale[j]*std::clamp(initial_raw[j],-10.f,10.f)):candidate[j]-candidates[j];
    double sum=0;int samples=0;bool failed=invalid(s);
    for(int step=0;step<steps&&!failed;step++) {
      if(step&&step%10==0) {
        std::copy(target,target+23,previous);
        future_target(f,s,history,velocity,phase,walking,previous,residual,candidate,goal,step*.002,target);
      }
      if(n==0&&step%10==0) {
        double* trace=f->debug_trace+(step/10)*82;
        std::copy(d->qpos,d->qpos+30,trace);std::copy(d->qvel,d->qvel+29,trace+30);std::copy(target,target+23,trace+59);
        f->debug_controls=step/10+1;
      }
      for(int j=0;j<23;j++) {
        double band=std::min(.1,.2*(s->limits[2*j+1]-s->limits[2*j]));
        double penetration=d->qpos[7+j]-std::clamp(d->qpos[7+j],s->limits[2*j]+band,s->limits[2*j+1]-band);
        double outward=penetration*d->qvel[6+j]>0?d->qvel[6+j]:0;
        d->ctrl[j]=std::clamp(s->kp[j]*(target[j]-d->qpos[7+j])-s->kd[j]*d->qvel[6+j]-100*penetration-2*outward,-s->effort[j],s->effort[j]);
      }
      mj_step(m,d);failed=invalid(s);
      if(!failed&&(step+1)%10==0){sum+=cost(s,goal,(step+1)*.002,initial_com);samples++;}
    }
    double movement=0;for(int j=0;j<23;j++)movement+=square((candidate[j]-candidates[j])/.25);
    scores[n]=failed?1e9:sum/std::max(1,samples)+.02*movement/23;
    std::copy(d->qpos,d->qpos+30,terminals+n*59);std::copy(d->qvel,d->qvel+29,terminals+n*59+30);
  }
  return int(std::min_element(scores,scores+count)-scores);
}
}
