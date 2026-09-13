"""Unchanged original-source CPU referee with the distinct frozen-base reader."""

from pathlib import Path

from gear_sonic.scripts import record_g1_true23_original_intent as recorder
from gear_sonic.utils import g1_true23_frozen_decoder_checkpoint as checkpoint
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def main():
    previous = recorder.load_cpu_actor

    def load_actor(*args, **kwargs):
        policy, identity, semantics = checkpoint.load_cpu_actor(*args, **kwargs)
        for path in (Path(__file__), Path(checkpoint.__file__)):
            path = path.resolve(strict=True)
            semantics["reverified_repository_sources"][str(path)] = sha256_file(path)
        return policy, identity, semantics

    recorder.load_cpu_actor = load_actor
    try:
        return recorder.main()
    finally:
        recorder.load_cpu_actor = previous


if __name__ == "__main__":
    main()
