def one_plan(planner, fresh_seed, args, seed_targets, data, warm_targets, completed):
    planner.window(completed + 10)
    if warm_targets is None:
        warm_targets = planner.target_reference(np.arange(args.horizon))
    else:
        warm_targets = np.concatenate((warm_targets[args.commit:], planner.target_reference(np.arange(args.horizon - args.commit, args.horizon))))
    actual_state = np.r_[data.qpos, data.qvel]
    tick = time.perf_counter()
    seed_selection = None
    cached_rollout = None
    if seed_targets is not None:
        seed_indices = np.minimum(completed + np.arange(args.horizon), len(seed_targets) - 1)
        recorded_targets = seed_targets[seed_indices]
        recorded_states, _, recorded_costs = planner.rollout(actual_state, recorded_targets)
        warm_states, _, warm_costs = planner.rollout(actual_state, warm_targets)
        use_recorded = recorded_costs[0] < warm_costs[0]
        seed_selection = dict(selected='recorded_bfm' if use_recorded else 'shifted_mpc', recorded_cost=float(recorded_costs[0]) if np.isfinite(recorded_costs[0]) else None, warm_cost=float(warm_costs[0]) if np.isfinite(warm_costs[0]) else None)
        if use_recorded:
            warm_targets = recorded_targets.copy()
            cached_rollout = (recorded_states[:, 0], recorded_costs[0])
        else:
            cached_rollout = (warm_states[:, 0], warm_costs[0])
    if fresh_seed is not None:
        if cached_rollout is None:
            warm_states, _, warm_costs = planner.rollout(actual_state, warm_targets)
            cached_rollout = (warm_states[:, 0], warm_costs[0])
            seed_selection = dict(selected='shifted_mpc', recorded_cost=None, warm_cost=float(warm_costs[0]) if np.isfinite(warm_costs[0]) else None)
        try:
            fresh_targets, fresh_diagnostics = fresh_seed.propose(completed, data.qpos, data.qvel, horizon=args.horizon)
        except BFMSeedRolloutError as error:
            seed_selection.update(fresh_cost=None, fresh_rejected=True, fresh_rejection=dict(message=str(error), diagnostics=error.diagnostics))
        else:
            fresh_states, _, fresh_costs = planner.rollout(actual_state, fresh_targets)
            seed_selection.update(fresh_cost=float(fresh_costs[0]) if np.isfinite(fresh_costs[0]) else None, fresh_diagnostics=fresh_diagnostics)
            if np.isfinite(fresh_costs[0]) and (not np.isfinite(cached_rollout[1]) or fresh_costs[0] < cached_rollout[1]):
                warm_targets = fresh_targets.copy()
                cached_rollout = (fresh_states[:, 0], fresh_costs[0])
                seed_selection['selected'] = 'fresh_bfm'
    planned_states, planned_targets, gains, cost = ilqr(planner, actual_state, warm_targets.copy(), iters=args.iterations, initial_rollout=cached_rollout)
    solve_ms = (time.perf_counter() - tick) * 1000
    return (planned_states, planned_targets, gains, cost, seed_selection, solve_ms)
