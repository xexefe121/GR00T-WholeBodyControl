// Native23 measured-state preview. No plant mutation, networking or file I/O.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <new>

struct Preview {
  const mjModel* model;
  mjData *seed, *probe[2];
  double kp[23], kd[23], effort[23], limits[46], previous[23];
  int steps, delay;
  double reserve;
  bool has_previous=false;
  int threads=2;
};

extern "C" {
void* preview_create(const mjModel* model,const double* kp,const double* kd,
    const double* effort,const double* limits,int steps,int delay,double reserve) {
  if (!model || model->nq!=30 || model->nv!=29 || model->nu!=23 ||
      std::abs(model->opt.timestep-.002)>1e-12 || steps<1 || steps>20 ||
      delay<0 || delay>6 || reserve<=0 || reserve>.001) return nullptr;
  auto* p=new(std::nothrow) Preview;
  if (!p) return nullptr;
  p->model=model;p->seed=mj_makeData(model);
  p->probe[0]=mj_makeData(model);p->probe[1]=mj_makeData(model);
  std::copy(kp,kp+23,p->kp);std::copy(kd,kd+23,p->kd);
  std::copy(effort,effort+23,p->effort);std::copy(limits,limits+46,p->limits);
  p->steps=steps;p->delay=delay;p->reserve=reserve;
  return p;
}
void preview_destroy(void* handle) {
  auto* p=static_cast<Preview*>(handle);
  if (p) {mj_deleteData(p->seed);mj_deleteData(p->probe[0]);mj_deleteData(p->probe[1]);delete p;}
}
int preview_parallelism() {return 2;}
int preview_set_parallelism(void* handle,int threads) {
  if (!handle || threads<1 || threads>2) return -1;
  static_cast<Preview*>(handle)->threads=threads;return threads;
}
void preview_reset(void* handle,const double* q,const double* v,const double* previous) {
  auto* p=static_cast<Preview*>(handle);
  mj_resetData(p->model,p->seed);
  std::copy(q,q+30,p->seed->qpos);std::copy(v,v+29,p->seed->qvel);
  mj_forward(p->model,p->seed);
  p->has_previous=previous!=nullptr;
  if (previous) std::copy(previous,previous+23,p->previous);
}
int preview_limits(void* handle,const double* target,double* low,double* high) {
  auto* p=static_cast<Preview*>(handle);
  const int schedules=p->delay && p->has_previous ? 2 : 1;
  double lows[2][23],highs[2][23];int invalid[2]{};
  #pragma omp parallel for num_threads(p->threads) if(p->threads>1)
  for (int schedule=0;schedule<schedules;++schedule) {
    const int delay=schedule?p->delay:0;
    mjData* data=p->probe[schedule];
    std::fill(lows[schedule],lows[schedule]+23,-std::numeric_limits<double>::infinity());
    std::fill(highs[schedule],highs[schedule]+23,-std::numeric_limits<double>::infinity());
    mj_copyData(data,p->model,p->seed);
    for (int step=0;step<p->steps+p->delay;++step) {
      const double* applied=step<delay?p->previous:target;
      for (int j=0;j<23;++j) {
        const double torque=p->kp[j]*(applied[j]-data->qpos[7+j])-p->kd[j]*data->qvel[6+j];
        data->ctrl[j]=std::clamp(torque,-p->effort[j],p->effort[j]);
      }
      mj_step(p->model,data);
      for (int j=0;j<23;++j) {
        const double q=data->qpos[7+j];
        if (!std::isfinite(q) || !std::isfinite(data->qvel[6+j])) invalid[schedule]=1;
        lows[schedule][j]=std::max(lows[schedule][j],p->limits[2*j]+p->reserve-q);
        highs[schedule][j]=std::max(highs[schedule][j],q-p->limits[2*j+1]+p->reserve);
      }
    }
  }
  if (invalid[0] || invalid[1]) return -1;
  for (int j=0;j<23;++j) {
    low[j]=schedules==2?std::max(lows[0][j],lows[1][j]):lows[0][j];
    high[j]=schedules==2?std::max(highs[0][j],highs[1][j]):highs[0][j];
  }
  return schedules;
}
}
