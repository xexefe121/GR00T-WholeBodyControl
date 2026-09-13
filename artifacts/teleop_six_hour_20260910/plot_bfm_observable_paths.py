"""Show full source-path/yaw mismatch and separate estimator error, without alignment."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import load_motion


def yaw(q):
    w,x,y,z=q.T
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))


def run():
    base=Path(__file__).resolve().parent/'bfm_observable_closed_loop_v1'
    results=[]
    for clip in ['pico','walk008']:
        case=base/f'{clip}_fixed_bias_noise'
        actual=np.load(case/'trace.npz')['qpos']
        motion,timeline,_=load_motion(clip)
        desired=np.c_[motion['body_pos_w'][10:10+len(actual),0],motion['body_quat_w'][10:10+len(actual),0]]
        phase=next(p for p in timeline['phases'] if p['name']=='source_motion')
        start,stop=phase['control_start']+1,phase['control_stop']+1
        ay,dy=yaw(actual[:,3:7]),yaw(desired[:,3:7])
        err=np.abs(np.angle(np.exp(1j*(ay-dy))))
        worst=start+int(np.argmax(err[start:stop]))
        source_time=(np.arange(start,stop)-start)/50.
        score=np.load(case/'privileged_odometry_score_only.npz')['error_position_start']
        sensor_t=np.load(case/'sensor_only_500hz.npz')['timestamp_s']
        fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
        for q,heading,label,color in [(desired,dy,'Desired reference','#087ca7'),(actual,ay,'Actual physics','#ca5819')]:
            axes[0].plot(q[start:stop,0],q[start:stop,1],label=label,color=color)
            axes[0].scatter(q[[start,stop-1],0],q[[start,stop-1],1],c=color,s=25)
            axes[1].plot(source_time,np.unwrap(heading)[start:stop]*180/np.pi,label=label,color=color)
        axes[0].set(xlabel='World X (m)',ylabel='World Y (m)',title='Source-phase root path',aspect='equal')
        axes[0].legend(fontsize=8)
        axes[1].set(xlabel='Source-phase elapsed (s)',ylabel='Unwrapped world yaw (degrees)',title='Heading: every source sample')
        axes[2].plot(sensor_t,np.linalg.norm(score[:,:2],axis=-1),color='#575a9a')
        axes[2].set(xlabel='Full lifecycle elapsed (s)',ylabel='Estimator XY error (m)',title='Estimator vs actual root (score only)')
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle(f'{clip.upper()} | Observable feedback + fixed sensor bias/noise | No path or heading alignment')
        path=case/'visual_comparison_v1/world_path_heading_and_odometry.png'
        path.parent.mkdir(exist_ok=True);fig.savefig(path,dpi=150);plt.close(fig)
        result=dict(clip=clip,worst_source_yaw_error_deg=float(err[worst]*180/np.pi),worst_qpos_index=worst,
            worst_timeline_s=worst/50.,worst_source_elapsed_s=(worst-start)/50.,image=str(path),no_path_alignment=True)
        results.append(result);print(json.dumps(result),flush=True)
    (base/'visual_path_receipt.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':run()
