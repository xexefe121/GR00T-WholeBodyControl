"""Unchanged full-request evaluation with additive exact inference-input traces."""

from types import FunctionType

from gear_sonic.scripts import evaluate_g1_true23_motion_ppo as evaluation
from gear_sonic.utils.g1_true23_policy_input_trace import run_recorded_case


def main():
    original = evaluation.main
    recorded = FunctionType(
        original.__code__,
        {**original.__globals__, "run_interior_case": run_recorded_case, "__file__": __file__},
        name=original.__name__,
        argdefs=original.__defaults__,
        closure=original.__closure__,
    )
    recorded.__kwdefaults__ = original.__kwdefaults__
    return recorded()


if __name__ == "__main__":
    raise SystemExit(main())
