"""Declared noise-scaled singular-value cap for saved native23 MPC feedback."""
import numpy as np


class NoiseCappedFeedback:
    def __init__(self, gains, *, noise_gain_cap=.04, correction_clip=.1):
        self.sigma=np.r_[np.full(3,.002),np.full(3,.001),np.full(23,.002),np.full(29,.01)]
        self.noise_gain_cap=float(noise_gain_cap);self.correction_clip=float(correction_clip)
        self.raw=np.asarray(gains,dtype=np.float64)
        if self.raw.ndim!=3 or self.raw.shape[1:]!=(23,58) or not np.isfinite(self.raw).all():
            raise ValueError("invalid native23 feedback gains")
        if not np.isfinite([noise_gain_cap,correction_clip]).all() or min(noise_gain_cap,correction_clip)<=0:
            raise ValueError("feedback caps must be finite and positive")
        left,singular,right=np.linalg.svd(self.raw*self.sigma,full_matrices=False)
        self.singular=singular
        self.gains=np.einsum("nij,nj,njk->nik",left,np.minimum(singular,noise_gain_cap),right)/self.sigma

    def correction(self,control,error):
        return np.clip(self.gains[control]@error,-self.correction_clip,self.correction_clip)

    def report(self):
        return dict(kind="noise_scaled_feedback_singular_value_cap",noise_std=self.sigma.tolist(),
                    noise_gain_cap_rad=self.noise_gain_cap,correction_clip_rad=self.correction_clip,
                    fraction_singular_values_capped=float(np.mean(self.singular>self.noise_gain_cap)),
                    raw_singular_p50_p95_p99_max=np.percentile(self.singular,[50,95,99,100]).tolist(),
                    time_smoothing=False,offline_gain_sequence_requires_saved_plan=True,
                    guaranteed_joint_noise_rms_cap_under_declared_independent_noise_rad=self.noise_gain_cap,
                    controller_stability_guaranteed=False)
