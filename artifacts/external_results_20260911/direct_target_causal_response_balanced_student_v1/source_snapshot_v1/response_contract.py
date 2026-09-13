"""One fixed warm causal continuation; no selected execution in this module."""
from context_contract import cosine_rate, COEFFICIENT

START_STEP = 68000
FINAL_STEP = 71000
UPDATES = 3000
OPTIMIZER_START = 3000
OPTIMIZER_FINAL = 6000
CHECKPOINT_SHA256 = '10d238198e2d7826c35eaa9bd4a13ff2e05050a7f601c1f212b585c4a90eaefd'
BUDGETS = dict(training_forward_rows=44058000, training_forward_calls=9000,
    training_updates=3000, diagnostic_Torch_rows=1470280, diagnostic_Torch_calls=5748,
    diagnostic_ORT_rows=367570, diagnostic_ORT_calls=1437,
    calibration_forward_calls=0, calibration_gradient_calls=0,
    native_calls=0, BFM_calls=0, manual_export_trace_calls=0)

def check_protocol(request):
    expected = dict(kind='causal_response_balanced_continuation', condition='causal',
        updates=UPDATES, ordinary_start_step=START_STEP, ordinary_final_step=FINAL_STEP,
        optimizer_start_step=OPTIMIZER_START, optimizer_final_step=OPTIMIZER_FINAL,
        fresh_optimizer=False, coefficient=COEFFICIENT, coefficient_recalibration=False,
        learning_rate=[1e-5,1e-6], weight_decay=1e-5, gradient_clip=10.,
        features=1323, context_order='previous_action23_then_incoming_history300',
        first_layer_execution='split_contiguous_1000_plus_323',
        export_first_layer_execution='monolithic_float64_1323',
        initial_parity_tolerance_rad=1e-5, initial_byte_gate_required=False,
        parity_tolerance_rad=1e-5, budgets=BUDGETS, automatic_retry=False,
        no_checkpoint_selection=True, context_and_normalization_reused=True)
    for key,value in expected.items():
        if request.get(key)!=value:raise ValueError('Fixed warm continuation differs: '+key)
    if request['subjects']['checkpoint']['sha256']!=CHECKPOINT_SHA256:
        raise ValueError('Exact causal68000 source checkpoint required.')

