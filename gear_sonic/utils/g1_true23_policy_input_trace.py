"""Record actual full-request inference without replacing controller behavior."""

import numpy as np

from gear_sonic.utils.g1_true23_interior_target_filter import run_interior_case


class RecordedPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.records = []

    def infer(self, encoder267, history930):
        inputs = {"encoder267": np.asarray(encoder267).copy(), "history930": np.asarray(history930).copy()}
        for key, size in (("encoder267", 267), ("history930", 930)):
            value = inputs[key]
            if value.shape != (size,) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("input recorder requires the original finite float32 policy boundary")
        record = {**inputs, "raw23": None, "decoder994": None, "error": None}
        self.records.append(record)
        try:
            # Preserve the exact arguments, return objects, and call count.
            result = self.policy.infer(encoder267, history930)
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        raw, decoder = result
        for value, size in ((raw, 23), (decoder, 994)):
            if value.shape != (size,) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("recorded policy returned a different or nonfinite boundary")
        np.testing.assert_array_equal(decoder[64:], history930)
        record.update(raw23=raw.copy(), decoder994=decoder.copy())
        return result

    def arrays(self):
        result = {}
        for key, size in (("encoder267", 267), ("history930", 930), ("raw23", 23), ("decoder994", 994)):
            result["policy_" + key] = np.asarray(
                [
                    np.full(size, np.nan, dtype=np.float32) if row[key] is None else row[key]
                    for row in self.records
                ],
                dtype=np.float32,
            ).reshape(-1, size)
        result["policy_inference_returned"] = np.asarray(
            [row["raw23"] is not None for row in self.records], dtype=np.bool_
        )
        return result


def run_recorded_case(**kwargs):
    options = dict(kwargs)
    recorder = RecordedPolicy(options["policy"])
    options["policy"] = recorder
    report, arrays = run_interior_case(**options)
    count, completed = len(recorder.records), report["completed_transitions"]
    if count not in (completed, completed + 1) or count > report["requested_transitions"]:
        raise ValueError("inference count differs from actual completed/terminal controller requests")
    if any(key.startswith("policy_") for key in arrays):
        raise ValueError("policy input recorder would replace existing trace data")
    arrays = {**arrays, **recorder.arrays()}
    report = {
        **report,
        "policy_input_trace": dict(
            kind="g1_true23_actual_inference_input_trace_v1",
            inference_calls=count,
            inference_returned=int(arrays["policy_inference_returned"].sum()),
            failed_inference=[
                dict(index=index, error=row["error"])
                for index, row in enumerate(recorder.records)
                if row["error"] is not None
            ],
            unavailable_outputs_are_nan_not_fabricated=True,
            original_policy_arguments_results_and_call_count_preserved=True,
            controller_history_or_reference_reconstructed=False,
            extra_physics_or_policy_calls=0,
            hardware_authorized=False,
            deployment_ready=False,
        ),
    }
    return report, arrays
