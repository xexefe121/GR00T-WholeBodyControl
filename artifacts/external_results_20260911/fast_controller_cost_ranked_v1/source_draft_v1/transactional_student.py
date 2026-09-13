"""Prepare one policy proposal, then commit exactly the selected actual action."""
import copy
import time
import numpy as np
from student_linear_runtime import FEATURES,infer_base


class TransactionalStudent:
    def __init__(self,runtime):
        self.runtime=runtime
        self.pending=None

    def _unchanged(self,pending):
        seed=self.runtime.seed
        return (seed.recorded_controls==pending['control'] and
            np.array_equal(seed.previous_action,pending['previous_action']) and
            seed.history.data.keys()==pending['history_before'].keys() and
            all(np.array_equal(seed.history.data[k],v) for k,v in pending['history_before'].items()))

    def begin(self,control,qpos,qvel,terminal=False,disable_head=False):
        if self.pending is not None:raise RuntimeError('Previous proposal has not been committed or cancelled.')
        runtime=self.runtime;seed=runtime.seed
        if control!=seed.recorded_controls:raise ValueError('Transactional policy control clock mismatch.')
        if terminal!=(control>=1269) or disable_head!=(control<250):raise ValueError('Transactional policy phase flags do not match the fixed lifecycle.')
        tick=time.perf_counter();previous=seed.previous_action.copy()
        before={key:value.copy() for key,value in seed.history.data.items()}
        _,terms=seed._terms(qpos,qvel,previous)
        next_history=copy.deepcopy(seed.history)
        history=next_history.before_update(terms)
        raw,base,sensed=infer_base(seed,qpos,qvel,previous,history,control+11,terminal)
        if terminal:
            features=np.zeros(FEATURES,np.float32);delta=np.zeros(23,np.float32)
        else:
            features=runtime.features(qpos,qvel,control+11,base,previous)
            delta=np.zeros(23,np.float32) if disable_head else runtime.head.run(None,{'features':features[None]})[0][0]
        if delta.shape!=(23,) or not np.isfinite(delta).all():raise ValueError('Invalid transactional head output.')
        target=np.clip(base+delta,runtime.limits[:,0],runtime.limits[:,1])
        proposed_action=(raw+delta*np.asarray(runtime.c['kp'])/(.25*np.asarray(runtime.c['training_effort']))).astype(np.float32)
        proposed=dict(target=target.copy(),base_target=base.copy(),delta=delta.copy(),previous_action=previous.copy(),
            action=proposed_action.copy(),state=sensed.copy(),history=history.copy(),features=features.copy(),
            raw_bfm_action=raw.copy(),inference_ms=(time.perf_counter()-tick)*1000)
        pending=dict(control=control,terminal=terminal,disable_head=disable_head,previous_action=previous,
            history_before=before,history_after=next_history,proposed=proposed)
        if not self._unchanged(pending):raise RuntimeError('Proposal generation changed actual policy history.')
        self.pending=pending
        return {key:value.copy() if isinstance(value,np.ndarray) else value for key,value in proposed.items()}

    def cancel(self):
        if self.pending is None:raise RuntimeError('No pending proposal to cancel.')
        if not self._unchanged(self.pending):raise RuntimeError('Policy state changed before proposal cancellation.')
        self.pending=None

    def commit(self,selected_target):
        pending=self.pending
        if pending is None:raise RuntimeError('No pending proposal to commit.')
        if not self._unchanged(pending):raise RuntimeError('Policy state changed before proposal commit.')
        runtime=self.runtime;seed=runtime.seed;selected=np.asarray(selected_target,np.float64)
        if selected.shape!=(23,) or not np.isfinite(selected).all():raise ValueError('Invalid selected target.')
        if np.any(selected<runtime.limits[:,0]) or np.any(selected>runtime.limits[:,1]):raise ValueError('Selected target violates native bounds.')
        proposed=pending['proposed'];primary_equal=np.array_equal(selected,proposed['target'])
        normalized=((selected-np.asarray(runtime.c['default_q']))*np.asarray(runtime.c['kp'])/(.25*np.asarray(runtime.c['training_effort']))).astype(np.float32)
        if pending['control']<250:
            if not pending['disable_head'] or pending['terminal'] or not primary_equal:raise ValueError('Initial BFM prefix must apply its unchanged primary proposal.')
            next_action=proposed['action'].copy()
        elif pending['terminal'] and primary_equal:
            next_action=proposed['raw_bfm_action'].copy()
        else:
            next_action=normalized.copy()
        if not np.isfinite(next_action).all():raise ValueError('Nonfinite selected action history.')
        values={key:value.copy() if isinstance(value,np.ndarray) else value for key,value in proposed.items()}
        values.update(target=selected.copy(),action=next_action.copy(),proposed_target=proposed['target'].copy(),
            proposed_action=proposed['action'].copy(),applied_normalized_action=normalized,
            applied_delta=selected-proposed['base_target'],primary_applied_unchanged=primary_equal)
        seed.history=pending['history_after'];seed.previous_action=next_action;seed.recorded_controls+=1
        self.pending=None
        return values
