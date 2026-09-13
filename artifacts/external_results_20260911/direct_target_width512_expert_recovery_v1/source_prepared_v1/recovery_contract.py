"""Fixed source-only recovery protocol and conservative work ceilings."""
START, SWITCH, LIFECYCLE, EXTENSION = 251, 1269, 1569, 250
HORIZON, ITERATIONS, COMMIT, SUBSTEPS = 30, 10, 5, 10
SOURCE_TRACE_SHA = 'b9061a1c9dc6aee65d713f16909a3135f072d94a5bc20af750b5958cc493629d'
SEMANTICS_SHA = '4a02cdd5d2f771a7fca5cb0cd002b9b4d5e3e121b4bbf3030f40181ebf2ed640'
SEMANTICS_OWNER_SHA = '7d9fee34c3bae9f41d0a133e3cbc8593fd8b3a3ac7415dc627adfca183290ebd'
HISTORY_SHAPES = {'actions': (4,23), 'base_ang_vel': (4,3), 'dof_pos': (4,23),
                  'dof_vel': (4,23), 'projected_gravity': (4,3)}
PLAN_CONTROLS = tuple(range(START, SWITCH, COMMIT))


def budgets():
    plans = len(PLAN_CONTROLS)
    restore = 2 * plans
    linearizations = (plans + restore) * ITERATIONS
    # Three normal seeds per window, one extra first-plan equality reroll,
    # feasible solver initial reroll plus <=10 line searches per normal solve,
    # <=two ordinary certificates per restoration-triggering window.
    ordinary_rollouts = 3*plans + 1 + plans*(1+ITERATIONS) + restore
    restoration_rollouts = restore*(1+ITERATIONS)
    terminal = LIFECYCLE-SWITCH+EXTENSION
    live = (LIFECYCLE-START+EXTENSION)*SUBSTEPS
    private = plans*HORIZON*SUBSTEPS + 3*HORIZON*SUBSTEPS + restore*HORIZON*SUBSTEPS + live
    rollout_count = ordinary_rollouts+restoration_rollouts
    bf = plans*HORIZON+terminal
    return {
        'ordinary_ilqr': (plans,plans), 'restoration_ilqr': (restore,restore),
        'ordinary_rollout': (ordinary_rollouts,ordinary_rollouts),
        'restoration_rollout': (restoration_rollouts,restoration_rollouts),
        'linearize': (linearizations,linearizations),
        'BFM_propose': (plans,plans), 'BFM_actor': (bf,bf), 'BFM_backward': (bf,8*bf),
        'native_live_step': (live,live), 'native_private_step': (private,private),
        'native_BFM_step': (plans*HORIZON*SUBSTEPS,plans*HORIZON*SUBSTEPS),
        'native_initial_certificate_step': (3*HORIZON*SUBSTEPS,3*HORIZON*SUBSTEPS),
        'native_restoration_certificate_step': (restore*HORIZON*SUBSTEPS,restore*HORIZON*SUBSTEPS),
        'native_preview_step': (live,live),
        'initial_native_certificate': (3,3), 'restoration_native_certificate': (restore,restore),
        'imminent_native_preview': (live//SUBSTEPS,live//SUBSTEPS),
        'batch_construct': (6,3*(HORIZON*(1+58+23)+9)),
        'batch_fd_step': (2*linearizations,linearizations*HORIZON*(1+58+23)*SUBSTEPS),
        'batch_line_step': (ordinary_rollouts*HORIZON*SUBSTEPS+restoration_rollouts*HORIZON,
                            rollout_count*HORIZON*9*SUBSTEPS),
        'batch_explicit_forward': (ordinary_rollouts*HORIZON+rollout_count+linearizations,
                                   ordinary_rollouts*HORIZON*9+rollout_count*9+linearizations*59),
    }


def protocol():
    return dict(initial_global_control=START, prefix_controls=START, original_BFM_prefix_controls=250,
        learned_prefix_controls=1, requested_branch_controls=LIFECYCLE-START,
        MPC_controls=SWITCH-START, terminal_BFM_controls=LIFECYCLE-SWITCH,
        conditional_hold_controls=EXTENSION, original_source_controls=819,
        requested_combined_main_controls=LIFECYCLE, actual_native_step_max=(LIFECYCLE-START+EXTENSION)*10,
        prefix_native_steps=START*10, combined_native_steps=18190,
        horizon=30, iterations=10, commit=5, batch_threads=2, BLAS_threads=1,
        first_planner_window_frame=261, first_output_frame=262,
        selected_query_controls=[START], plan_controls=list(PLAN_CONTROLS),
        final_MPC_commit=SWITCH-PLAN_CONTROLS[-1],
        branch_clock='saved repeated+.002 through2510, then continued unchanged',
        private_native_initialization='qualified BFM/Batch planning semantics; never substitutes actual full291',
        native_tolerances=dict(joint_bound_excess_rad=1e-6,effort_ratio=1+1e-9,speed_ratio=1.,clock_seconds=1e-10),
        budgets={k:dict(attempted_calls_max=v[0],attempted_units_max=v[1]) for k,v in budgets().items()},
        batch_unit='sum of selected lanes times nstep; not actual plant steps',
        native_private_aggregate='sum of disjoint BFM, initial-certificate, restoration-certificate and imminent-preview categories',
        forward_unit='explicit Batch.forward selected lanes; implicit Batch construction/step forwarding is not a separate wrapped Python call',
        BFM_backward_unit='input state rows; maximum8 per call',
        labels_admissible=False, additional_query_selection=False, model_training_authorized=False,
        hardware_authorized=False, realtime_policy_qualification=False)
