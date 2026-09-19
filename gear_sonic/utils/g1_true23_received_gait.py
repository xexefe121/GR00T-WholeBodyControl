"""Causal phase servo for the factory policy's exposed gait-clock inputs.

This experimental adapter consumes only the receiver's owned pose and backward
velocity. It does not select a phase from a clip index or a prepared schedule.
"""
import math
import numpy as np

class ReceivedGaitPhase:
    def __init__(self,frequency_bounds=(1.,1.2,1.3)):
        self.minimum,self.base,self.maximum=map(float,frequency_bounds)
        self.frequency=self.base;self.cadence=self.base;self.observed=None;self.previous_observed=None
        self.motion=False;self.confidence=0.;self.last_correction=0.

    def observe(self,reference):
        q=reference['joint'][-1];dq=reference['joint_velocity'][-1]
        feet=reference['feet'][-1];fv=reference['feet_velocity'][-1]
        omega=2*math.pi*max(.4,self.frequency)
        # Native factory hip difference is approximately +sin(phase), while
        # left-minus-right foot height is -sin(phase). Derivatives provide the
        # quadrature component. Combine dimensionless received leg signals.
        signal=complex((dq[0]-dq[6])/(omega*.25)-(fv[0,2]-fv[1,2])/(omega*.04),
            (q[0]-q[6])/.25-(feet[0,2]-feet[1,2])/.04)
        self.confidence=abs(signal)
        self.observed=(math.atan2(signal.imag,signal.real)/(2*math.pi))%1 if self.confidence>.25 else None
        self.motion=bool(np.max(np.abs(dq[:12]))>.1 or np.max(np.abs(fv))>.04)

    @staticmethod
    def difference(a,b):return (a-b+.5)%1-.5

    def advance(self,phase,walking):
        self.last_correction=0.
        if not walking:
            self.previous_observed=None;self.frequency=self.base;self.cadence=self.base
            return 0.,0.
        if self.observed is not None:
            if self.previous_observed is not None:
                received_rate=self.difference(self.observed,self.previous_observed)/.02
                if 0<=received_rate<=3:
                    self.cadence=float(np.clip(.95*self.cadence+.05*received_rate,self.minimum,self.maximum))
            self.previous_observed=self.observed
            # Correct through the vendor's supported frequency interval. The
            # phase advance and the frequency sent to the network must agree.
            # Never jump phase independently of the encoded gait frequency.
            self.frequency=float(np.clip(self.cadence+2*self.difference(self.observed,phase),self.minimum,self.maximum))
        else:
            self.previous_observed=None
            self.frequency=self.cadence
        self.last_correction=(self.frequency-self.base)*.02
        phase=(phase+self.last_correction)%1
        return phase,self.frequency

    def state(self):return self.__dict__.copy()
    def restore(self,state):self.__dict__.update(state)
