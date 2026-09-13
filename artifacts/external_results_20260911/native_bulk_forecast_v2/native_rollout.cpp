// Private fixed-horizon manual-PD rollout. No policy or acceptance decision.
#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdint>
#include <cstring>

static_assert(sizeof(mjtNum) == 8 && sizeof(int) == 4 && mjNWARNING == 8);

extern "C" int sonic_native_version() { return mj_version(); }
extern "C" void* sonic_step_address() { return reinterpret_cast<void*>(&mj_step); }

extern "C" int sonic_private_rollout(
    const mjModel* m, mjData* d, const double* target, const double* kp,
    const double* kd, const double* effort, int steps,
    double* q, double* v, double* times, double* torque, double* force,
    int* warnings, int* infos) {
  if (!m || !d || mj_version() != 323 || m->nq != 30 || m->nv != 29 ||
      m->nu != 23 || m->njnt != 24 || m->opt.timestep != .002 ||
      steps < 1 || steps > 50) return -1;
  auto save = [&](int s) {
    std::memcpy(q + 30*s, d->qpos, 30*sizeof(double));
    std::memcpy(v + 29*s, d->qvel, 29*sizeof(double));
    times[s] = d->time;
    for (int i = 0; i < 8; ++i) {
      warnings[8*s+i] = d->warning[i].number;
      infos[8*s+i] = d->warning[i].lastinfo;
    }
  };
  save(0);
  for (int s = 1; s <= steps; ++s) {
    for (int j = 0; j < 23; ++j) {
      // Each operation rounds independently: compile with contraction OFF.
      double difference = target[j] - d->qpos[7+j];
      double proportional = difference * kp[j];
      double damping = d->qvel[6+j] * kd[j];
      double command = proportional - damping;
      double lower = -effort[j];
      command = command < lower ? lower : command;
      command = command > effort[j] ? effort[j] : command;
      d->ctrl[j] = command;
      torque[23*(s-1)+j] = command;
    }
    mj_step(m, d);
    save(s);
    std::memcpy(force + 23*(s-1), d->qfrc_actuator+6, 23*sizeof(double));
    // Critical engine/nonfinite stop only; all strict acceptance is Python.
    bool critical = !std::isfinite(d->time);
    for (int j = 0; j < 30; ++j) critical |= !std::isfinite(d->qpos[j]);
    for (int j = 0; j < 29; ++j) critical |= !std::isfinite(d->qvel[j]);
    for (int j = 0; j < 23; ++j) critical |= !std::isfinite(d->qfrc_actuator[6+j]);
    for (int j = 0; j < 8; ++j) critical |= d->warning[j].number != 0;
    if (critical) return s;
  }
  return steps;
}
