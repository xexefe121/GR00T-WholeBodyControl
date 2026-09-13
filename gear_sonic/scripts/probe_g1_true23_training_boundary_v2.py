"""Correct only the diagnostic reader's explicit 268-to267 route handling.

The strict CPU reader is constructed with a267 semantic dummy; training uses
268 (one route index plus267 semantic values). The inherited actor flag is
inferred from that construction shape, not a checkpoint tensor. Restore the
training-shaped dispatch before invoking it on actual observation groups.
No weights, recorded arrays, observation function, physics or limit is changed.
"""

from pathlib import Path

from gear_sonic.scripts import probe_g1_true23_training_boundary as probe
from gear_sonic.utils import g1_true23_bounded_progress_checkpoint as checkpoint
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_training_precision import ieee_training_precision


def main():
    if sha256_file(Path(probe.__file__)) != "455a9dec35adb5510066f932daf9b408ba5d4aed4e954fe38bb9e85a3cd56e10":
        raise ValueError("original failed boundary probe changed")
    previous = checkpoint.load_cpu_actor

    def training_shape_reader(*args, **kwargs):
        reader, identity, semantics = previous(*args, **kwargs)
        if reader.actor.tokenizer_has_encoder_index is not False:
            raise ValueError("CPU reader no longer has the diagnosed267 construction shape")
        reader.actor.tokenizer_has_encoder_index = True
        identity["diagnostic_dispatch"] = dict(
            actual_executor="same_actor_CUDA_IEEE_float32_batch32",
            original_CPU_reader_constructed_semantic_dim=267,
            actual_training_group_dim=268,
            leading_training_route_index_removed_before_frozen_encoder=True,
            checkpoint_tensors_changed=False,
            original_CPU_inference_method_changed=False,
        )
        path = Path(__file__).resolve(strict=True)
        semantics["reverified_repository_sources"][str(path)] = sha256_file(path)
        return reader, identity, semantics

    checkpoint.load_cpu_actor = training_shape_reader
    try:
        with ieee_training_precision() as (precision, guard):
            probe.main(precision, guard)
    finally:
        checkpoint.load_cpu_actor = previous


if __name__ == "__main__":
    main()
