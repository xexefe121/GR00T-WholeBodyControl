"""Frozen BFM goal method; sole numerical change is terminal yaw gain2 to4."""
import numpy as np
from gear_sonic.utils.g1_true23_bfm_seed_observations import _quaternion_matrix

def terminal_goal_yaw4(self, frame, qpos):
    frame = min(frame, len(self.state)-1)
    stop = min(frame+8, len(self.state))
    states = self.state[frame:stop].copy(); priv = self.privileged[frame:stop].copy()
    refrot = _quaternion_matrix(self.motion["body_quat_w"][frame,0]); actrot = _quaternion_matrix(qpos[3:7])
    ry = np.arctan2(refrot[1,0],refrot[0,0]); ay = np.arctan2(actrot[1,0],actrot[0,0])
    dy = np.arctan2(np.sin(ry-ay),np.cos(ry-ay))
    omega = np.array([0.,0.,np.clip(4*dy,-.8,.8)])
    delta = self.motion["body_pos_w"][frame,0]-qpos[:3]; delta[2] = 0
    delta *= 3.0  # Controller position feedback gain; reference remains unchanged.
    delta *= min(1.,.6/max(np.linalg.norm(delta),1e-8))
    heading_actual = np.array([[np.cos(ay),-np.sin(ay),0],[np.sin(ay),np.cos(ay),0],[0,0,1.]])
    for t,index in enumerate(range(frame,stop)):
        rot = _quaternion_matrix(self.motion["body_quat_w"][index,0]); yaw = np.arctan2(rot[1,0],rot[0,0])
        heading = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
        positions = np.vstack((np.zeros(3),priv[t,1:73].reshape(24,3)))
        velocity = self.motion["body_lin_vel_w"][index,0]
        local_delta = heading_actual.T@(velocity+delta)-heading.T@velocity
        priv[t,223:298] += (local_delta+np.cross(omega,positions)).reshape(-1)
        priv[t,298:373] += np.tile(omega,25)
        states[t,-3:] += omega
    latent = self.sessions["backward"].run(None,dict(state=np.ascontiguousarray(states,dtype=np.float32),
                                                     privileged=np.ascontiguousarray(priv,dtype=np.float32)))[0].mean(0,keepdims=True)
    return 16*latent/np.maximum(np.linalg.norm(latent,axis=-1,keepdims=True),1e-12)
