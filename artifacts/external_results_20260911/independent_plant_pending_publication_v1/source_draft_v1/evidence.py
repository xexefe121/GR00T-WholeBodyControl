"""Owned postrun export schema, pure copies only. No additional native snapshots."""
from dataclasses import asdict
import numpy as np
from clock_core import digest


def _stack(records,field,dtype,width):
    if not records:return np.empty((0,width),dtype)
    values=[np.frombuffer(getattr(record,field),dtype) for record in records]
    if any(value.shape!=(width,) for value in values):raise ValueError('recorded byte schema differs: '+field)
    return np.stack(values).copy()


def export_owned(session):
    """Preserve complete/partial committed evidence; no success claim or disk I/O."""
    plant=session.foundation;steps=plant.step_records.records();controls=plant.control_records.records()
    captures=[r.captured for r in steps]
    if any(len(r.state)!=373*8 for r in captures):raise ValueError('native capture must remain373 float64')
    packed=np.stack([np.frombuffer(r.state,np.float64) for r in captures]).copy() if captures else np.empty((0,373),np.float64)
    arrays=dict(packed_capture=packed,step_integration=packed[:,:291].copy(),step_qpos=packed[:,291:321].copy(),
        step_qvel=packed[:,321:350].copy(),step_actuator_force=packed[:,350:373].copy(),
        step_command=_stack(captures,'torque',np.float64,23),step_target=_stack(steps,'target',np.float64,23),
        step_raw_action=_stack(steps,'raw_action',np.float32,23),
        step_warning_counts=np.asarray([r.warnings.counts for r in captures],np.int32).reshape(-1,8),
        step_warning_lastinfo=np.asarray([r.warnings.lastinfo for r in captures],np.int32).reshape(-1,8),
        control_integration=_stack(controls,'snapshot_state',np.float64,291),
        control_incoming_raw=_stack(controls,'incoming_raw',np.float32,23),
        control_flat_history_before=_stack(controls,'flat_history_before',np.float32,300))
    for field in ['index','nominal_start','nominal_end','actual_start','actual_end','wake_lateness_ns','finish_lateness_ns','wake_debt']:
        arrays['step_'+field]=np.asarray([getattr(r,field) for r in steps],np.int64)
    arrays['step_expected_simulation_time']=np.asarray([r.expected_simulation_time for r in steps],np.float64)
    arrays['step_verified']=np.asarray([r.verified for r in steps],np.bool_)
    for field in ['control','physics','nominal_activation_ns','actual_activation_ns','admitted_ns']:
        arrays['control_'+field]=np.asarray([getattr(r,field) for r in controls],np.int64)
    arrays['control_held']=np.asarray([r.held for r in controls],np.bool_)
    arrays['control_nominal_source_frame']=np.asarray([session.table.frames[r.control] for r in controls],np.int64)
    windows={value:index for index,value in enumerate(session.table.windows)}
    arrays['control_active_source_frame']=np.asarray([session.table.frames[windows[r.active_window_id]] for r in controls],np.int64)
    arrays['control_target']=np.stack([r.command.values() for r in controls]) if controls else np.empty((0,23),np.float64)
    arrays['control_raw_action']=np.stack([np.frombuffer(r.command.raw_action,np.float32) for r in controls]).copy() if controls else np.empty((0,23),np.float32)
    boundary=session.main_controls
    continuity=dict(boundary_control=boundary,present=len(controls)>boundary,no_reset_in_scaffold=True)
    if continuity['present']:
        current=controls[boundary];prior=controls[boundary-1]
        last=steps[boundary*10-1]
        continuity.update(full291_matches_previous_step=current.snapshot_state==last.captured.state[:291*8],
                          history_matches_previous_control=current.history_before==prior.history_after,
                          incoming_matches_previous_actual_action=current.incoming_raw==prior.command.raw_action,
                          full291_sha256=digest(current.snapshot_state),history_digest=digest(current.flat_history_before))
    raw_capsules={name:value for name,value in {
        'last_foundation_capture_return':plant.last_capture_return,'last_foundation_verifier_return':plant.last_verifier_return,
        'outer_driver_failure':session.driver_failure,'step_ledger_overflow':None,
        'transport_ledger_overflow':session.transport.overflow_record,'event_ledger_overflow':plant.events.overflow_record,
        'outer_ledger_overflow':session.outer.overflow_record}.items() if value is not None}
    fault=getattr(session.stepper,'failure',None)
    if fault is not None:
        raw_capsules['native_fault_evidence']=fault.evidence
        if fault.capture_return_evidence is not None:raw_capsules['native_capture_return_evidence']=fault.capture_return_evidence
    # A validated capture can exist even if its StepRecord failed to commit.
    last=plant.last_capture
    if last is not None:
        raw_capsules['last_validated_capture_state']=last.state
        raw_capsules['last_validated_capture_torque']=last.torque
    metadata=dict(summary=session.summary(),initial_time=session.stepper.initial_time,
        step_command_ids=[r.command_id for r in steps],step_issues=[r.issue for r in steps],
        control_command_ids=[r.command.command_id for r in controls],
        control_nominal_windows=[r.nominal_window_id for r in controls],control_active_windows=[r.active_window_id for r in controls],
        control_history_before=[r.history_before for r in controls],control_history_after=[r.history_after for r in controls],
        control_measured_terms=[r.terms for r in controls],
        step_ledger_overflow=plant.step_records.overflow_record,control_ledger_overflow=plant.control_records.overflow_record,
        transport_records=session.transport.records(),foundation_events=plant.events.records(),outer_cycles=session.outer.records(),
        hold_continuity=continuity,root_saved_array_acceptance_pending=True)
    initial=getattr(session.stepper,'initial_capture',None)
    if initial is not None:
        raw_capsules['initial_capture_state']=initial.state
        raw_capsules['initial_capture_torque']=initial.torque
        metadata['initial_warning_ledger']=asdict(initial.warnings)
    return arrays,metadata,raw_capsules
