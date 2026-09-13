"""Small literal saved-row regression plus independent positive-sum rounding proof."""
from fractions import Fraction
from pathlib import Path
import ast
import numpy as np

CELLS=np.array([0.00029250048100948334,0.00040344620356336236,0.0002317903854418546,
    0.0002385824773227796,0.0003928158839698881,0.00023211189545691013,0.00026231512310914695,
    0.00043314433423802257,0.00018598328460939229,0.00015109764353837818,0.0014195428229868412,
    0.00012657931074500084,0.00015321197861339897,0.0002670146059244871,9.537549340166152e-05],np.float32)
SAVED=np.float64(0.00032570085022598505)

def test_saved_update191_satisfies_original_mean_contract():
    reference=CELLS.astype(np.float64).mean();cpu32=np.float64(CELLS.mean())
    assert np.isclose(SAVED,reference,rtol=3e-7,atol=1e-14)
    assert not np.isclose(SAVED,cpu32,rtol=3e-7,atol=1e-14)
    assert (SAVED-cpu32)/np.spacing(np.float32(cpu32))==4
    assert not np.isclose(SAVED*1.00001,reference,rtol=3e-7,atol=1e-14)

def test_exact_positive_mean_and_analytical_rounding_bound():
    exact=sum((Fraction.from_float(float(v)) for v in CELLS),Fraction())/15
    unit=Fraction(1,2**24);gamma16=(16*unit)/(1-16*unit)
    error=abs(Fraction.from_float(float(SAVED))-exact)
    assert all(v>0 for v in CELLS) and error<=gamma16*exact
    assert error/exact<Fraction(3,10**7)
    assert float(exact)==CELLS.astype(np.float64).mean()

def test_original_nominal_gate_and_threshold_still_present():
    path=Path(__file__).with_name('audit_saved_warm.py');source=path.read_text();tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='close']
    nominal=next(n for n in calls if n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='nominal_cells')
    assert ast.unparse(nominal.args[2])=="cell_values['nominal'].mean(axis=1)"
    assert any(k.arg=='nominal' and isinstance(k.value,ast.Constant) and k.value.value is True for k in nominal.keywords)
    assert 'rtol=3e-7 if nominal else 5e-12' in source and 'nominal_f32_mean' not in source

def test_two_positive_float32_reduction_orders_need_not_match_each_other():
    values=np.array([.0010373692493885756,.0008217864669859409,.000872352859005332,.004755796864628792,
        .005181534215807915,.003716705832630396,.0036157453432679176,.0015894613461568952,.00023257006250787526,
        .0007430805126205087,.000698481104336679,.0066577172838151455,.002342177089303732,.0005448181182146072,.0016970483120530844],np.float32)
    sequential=np.float32(0)
    for value in values:sequential=np.float32(sequential+value)
    sequential=np.float32(sequential/np.float32(15));numpy=values.mean();reference=values.astype(np.float64).mean()
    assert np.isclose(sequential,reference,rtol=3e-7,atol=1e-14) and np.isclose(numpy,reference,rtol=3e-7,atol=1e-14)
    assert not np.isclose(sequential,numpy,rtol=3e-7,atol=1e-14)
