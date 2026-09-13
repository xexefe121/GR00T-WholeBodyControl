"""Comparison-only gates for the first changed feedback; no state injection."""
import numpy as np
from clipped_feedback import clipped_component_feedback

ORIGINAL_TRACE_SHA='38428ae27055be7c2e771c5a23056c858f004ef38f260a0b45a57662d8d0aae3'
CONTROL_FIELDS=('target','source_frame','global_control','controller_mode','joint_error','root_error',
    'state','history','previous_action','action','base_target','delta','features',
    'physics_substeps','control_integration_before','control_history_before','control_previous_action_before',
    'raw_proposal','actual_normalized_action')
SAMPLE_FIELDS=('physics_qpos','physics_qvel','physics_time','physics_expected_time',
    'physics_warning_counts','physics_warning_lastinfo')
STEP_FIELDS=('physics_torque','physics_actuator_torque','range_excess','velocity_ratio','effort_ratio','clock_error')


class FeedbackParity:
    def __init__(self,original,contract):
        self.original=original;self.contract=contract
        clips=np.any(original['raw_proposal'][250:]!=original['target'][250:],axis=1)
        if not np.any(clips) or int(np.flatnonzero(clips)[0])+250!=264:
            raise ValueError('Bound original first learned clipping must be control264.')
        self.first_feedback,self.first_native_mask,self.first_mask=clipped_component_feedback(
            original['action'][264],original['raw_proposal'][264],original['target'][264],contract,True)
        self.lag266=original['history'][266].copy()
        # Sorted history begins with four 23-component actions, newest first.
        self.lag266[:23]=self.first_feedback

    def before(self,control,trace,integration,qpos,qvel,history,previous,count,warnings,lastinfo,expected_time):
        old=self.original
        if control==265:
            pairs=[(key,np.asarray(trace[key])[:265],old[key][:265]) for key in CONTROL_FIELDS]
            pairs.extend((key,np.asarray(trace[key]),old[key][:266]) for key in ('qpos','qvel'))
            pairs.extend((key,np.asarray(trace[key]),old[key][:2651]) for key in SAMPLE_FIELDS)
            pairs.extend((key,np.asarray(trace[key]),old[key][:2650]) for key in STEP_FIELDS)
            pairs.extend([
                ('raw_combined_actions',np.asarray(trace['raw_combined_action']),old['action'][:265]),
                ('feedback_before_first_clip',np.asarray(trace['feedback_action'])[:264],old['action'][:264]),
                ('feedback_at264',np.asarray(trace['feedback_action'])[264],self.first_feedback),
                ('feedback_mask_at264',np.asarray(trace['feedback_clip_mask'])[264],self.first_mask),
                ('no_earlier_feedback_mask',np.asarray(trace['feedback_clip_mask'])[:264],np.zeros((264,23),bool)),
                ('integration_at265',integration,old['control_integration_before'][265]),
                ('qpos_at265',qpos,old['qpos'][265]),('qvel_at265',qvel,old['qvel'][265]),
                ('history_before265',history,old['control_history_before'][265]),
                ('previous_action_at265',previous,self.first_feedback),
                ('recorded_controls_at265',np.asarray(count,dtype=np.int64),np.asarray(265,dtype=np.int64)),
                ('warnings_at265',warnings,old['physics_warning_counts'][2650]),
                ('warning_lastinfo_at265',lastinfo,old['physics_warning_lastinfo'][2650]),
                ('expected_clock_at265',np.asarray(expected_time),old['physics_expected_time'][2650])])
            return 'original_prefix265_and_first_feedback_parity.json',pairs
        if control==266:
            return 'first_feedback_lag_history266_parity.json',[
                ('history_at266',history,self.lag266),
                ('previous_action_at266',previous,np.asarray(trace['feedback_action'])[-1]),
                ('recorded_controls_at266',np.asarray(count,dtype=np.int64),np.asarray(266,dtype=np.int64))]
        return None

    def after(self,control,proposed):
        if control==265:
            return 'first_feedback_actor_input265_parity.json',[
                ('previous_action',proposed['previous_action'],self.first_feedback),
                ('unchanged_actor_lag_history',proposed['history'],self.original['history'][265]),
                ('unchanged_measured_state',proposed['state'],self.original['state'][265])]
        if control==266:
            return 'first_feedback_actor_lag266_parity.json',[
                ('actor_lag_history',proposed['history'],self.lag266)]
        return None
