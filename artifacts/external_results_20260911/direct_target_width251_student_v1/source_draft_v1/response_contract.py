"""Fixed width512 warm protocol; actual launch requires separate root clearance."""
from context_contract import COEFFICIENT
from width512 import proposed_rate as cosine_rate

START_STEP = 71000
FINAL_STEP = 81000
UPDATES = 10000
OPTIMIZER_START = 6000
OPTIMIZER_FINAL = 16000
CHECKPOINT_SHA256 = '395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d'
BUDGETS = dict(training_forward_rows=146860000, training_forward_calls=30000,
    training_updates=10000, diagnostic_Torch_rows=1470280, diagnostic_Torch_calls=5748,
    diagnostic_ORT_rows=367570, diagnostic_ORT_calls=1437,
    calibration_forward_calls=0, calibration_gradient_calls=0,
    native_calls=0, BFM_calls=0, manual_export_trace_calls=0)

def check_protocol(request):
    expected = dict(kind='causal_width512_warm_continuation', condition='causal',
        updates=UPDATES, ordinary_start_step=START_STEP, ordinary_final_step=FINAL_STEP,
        optimizer_start_step=OPTIMIZER_START, optimizer_final_step=OPTIMIZER_FINAL,
        fresh_optimizer=False, coefficient=COEFFICIENT, coefficient_recalibration=False,
        learning_rate=dict(ramp_updates=250,ramp_inclusive=[1e-6,1e-5],cosine_updates=9750,cosine_inclusive=[1e-5,1e-6]),
        weight_decay=1e-5, gradient_clip=10.,
        features=1323,architecture=[1323,512,512,23],old_hidden_width=256,new_hidden_width=512,expansion_seed=20260912,
        context_order='previous_action23_then_incoming_history300',
        first_layer_execution='split_old256_new256_original1000_plus323',
        export_first_layer_execution='monolithic_float64_1323',
        initial_parity_tolerance_rad=1e-5, initial_byte_gate_required=False,
        parity_tolerance_rad=1e-5, budgets=BUDGETS, automatic_retry=False,
        no_checkpoint_selection=True, context_and_normalization_reused=True,
        old_optimizer_blocks_exact=True,new_optimizer_moments_zero=True,shared_optimizer_step_for_new_entries=6000)
    for key,value in expected.items():
        if request.get(key)!=value:raise ValueError('Fixed width512 protocol differs: '+key)
    if request['subjects']['checkpoint']['sha256']!=CHECKPOINT_SHA256:
        raise ValueError('Exact causal71000 source checkpoint required.')
