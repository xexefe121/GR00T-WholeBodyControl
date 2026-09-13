"""Protocol inputs are unselected until a future concrete request is admitted."""
from dataclasses import dataclass
import math
from balance_contract import GROUP_WEIGHTS

START_STEP=81000
OPTIMIZER_START=16000
COEFFICIENT=1.8188207859141674
SOURCE_CHECKPOINT='825f86468fcc83913066a9a818f4f438d080e054e17aa4aaadc1451974b8688e'

@dataclass(frozen=True)
class Protocol:
    updates:int
    rates:tuple
    recovery_coefficient:float

    @property
    def final_step(self):return START_STEP+self.updates
    @property
    def optimizer_final(self):return OPTIMIZER_START+self.updates
    @property
    def budgets(self):
        return dict(training_forward_rows=15704*self.updates,training_forward_calls=4*self.updates,training_updates=self.updates,
            diagnostic_Torch_rows=4*368588,diagnostic_Torch_calls=4*1441,diagnostic_ORT_rows=368588,diagnostic_ORT_calls=1441,
            calibration_forward_calls=0,calibration_gradient_calls=0,native_calls=0,BFM_calls=0,manual_export_trace_calls=0)

    def rate(self,index):
        if type(index) is not int or not 0<=index<self.updates:raise ValueError('Fixed selected update index')
        return self.rates[index]

def protocol(request):
    updates=request['updates'];rates=request['learning_rate_values'];coefficient=request['recovery_coefficient']
    if type(updates) is not int or not 1<=updates<=10000:raise ValueError('Use only a selected prefix of the existing10000 schedule')
    if not isinstance(rates,list) or len(rates)!=updates or any(type(v) is not float or not math.isfinite(v) or v<=0 for v in rates):
        raise ValueError('One explicitly frozen positive learning rate for each selected update required')
    if type(coefficient) is not float or not math.isfinite(coefficient) or coefficient<=0:raise ValueError('Positive explicitly selected recovery coefficient')
    return Protocol(updates,tuple(rates),coefficient)

def check_protocol(request):
    p=protocol(request)
    expected=dict(kind='qualified_width251_recovery_warm512_fit',condition='causal',ordinary_start_step=START_STEP,
        ordinary_final_step=p.final_step,optimizer_start_step=OPTIMIZER_START,optimizer_final_step=p.optimizer_final,
        fresh_optimizer=False,coefficient=COEFFICIENT,coefficient_recalibration=False,weight_decay=1e-5,gradient_clip=10.,
        features=1323,architecture=[1323,512,512,23],context_order='previous_action23_then_incoming_history300',
        first_layer_execution='split_old256_new256_original1000_plus323',export_first_layer_execution='monolithic_float64_1323',
        initial_parity_tolerance_rad=1e-5,initial_byte_gate_required=False,parity_tolerance_rad=1e-5,
        automatic_retry=False,no_checkpoint_selection=True,context_and_normalization_reused=True,
        expansion_performed=False,recovery_rows=1018,recovery_phase_counts=[99,819,100],
        recovery_objective='equal_three_phase_normalized_MSE',response_schedule='unchanged10000_prefix_no_wrap',
        group_weights=list(GROUP_WEIGHTS),consistency_evidence_reviewed=True,budgets=p.budgets)
    for name,value in expected.items():
        if type(request.get(name)) is not type(value) or request[name]!=value:raise ValueError('Selected recovery protocol differs: '+name)
    if request['subjects']['checkpoint']['sha256']!=SOURCE_CHECKPOINT:raise ValueError('Exact ordinary81000 source required')
    return p
