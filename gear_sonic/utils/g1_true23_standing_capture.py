"""Received-only standing capture shared by single and batched control."""
import numpy as np


def standing_capture(reference,qpos,qvel,standing_legs,motion_seen,stationary_since,now):
    """All inputs have a leading world axis. Missing dwell uses NaN."""
    r=reference;q=np.asarray(qpos);v=np.asarray(qvel);now=np.broadcast_to(now,(len(q),))
    error=np.max(np.abs(r['joint'][:,:12]-standing_legs),axis=1)
    seen=np.asarray(motion_seen)| (error>.25)
    stationary=(np.max(np.abs(r['joint_velocity'][:,:12]),axis=1)<.005)
    stationary &= np.linalg.norm(r['root_velocity'],axis=1)<.005
    stationary &= np.linalg.norm(r['root_omega'],axis=1)<.005
    stationary &= np.max(np.abs(r['feet_velocity']),axis=(1,2))<.005
    stationary &= seen&(error<.15)
    since=np.where(stationary,np.where(np.isnan(stationary_since),now,stationary_since),np.nan)
    w,x,y,z=q[:,3:7].T
    yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    goal=np.arctan2(r['root_rotation'][:,1,0],r['root_rotation'][:,0,0])
    yaw_error=np.arctan2(np.sin(goal-yaw),np.cos(goal-yaw))
    capture=(~np.isnan(since))&(now-since>=.2)&(np.linalg.norm(r['root'][:,:2]-q[:,:2],axis=1)<.05)
    capture &= (np.abs(yaw_error)<np.deg2rad(5))&(np.arccos(np.clip(1-2*(x*x+y*y),-1,1))<.15)
    capture &= np.linalg.norm(v[:,:2],axis=1)<.25
    return seen,since,capture
