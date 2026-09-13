"""Versioned metadata-only repair of the preserved raw-history recorder v1."""

from pathlib import Path

from gear_sonic.scripts import record_g1_true23_source_raw_history as previous
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


class MetadataCompatibleAdapter(previous.SourceRawHistoryAdapter):
    def contract(self):
        result = super().contract()
        # The unchanged referee copies this field into its result after physics.
        # Preserve the explicit new history semantics; do NOT restore the old
        # target-equivalent history label merely to satisfy its result schema.
        result["source_action_codec"]["previous_action"] = result["previous_action_counterfactual"][
            "previous_action"
        ]
        return result


def main():
    old_adapter, old_load = previous.SourceRawHistoryAdapter, previous.load_cpu_actor

    def load(*args, **kwargs):
        policy, identity, semantics = old_load(*args, **kwargs)
        path = Path(__file__).resolve(strict=True)
        semantics["reverified_repository_sources"][str(path)] = sha256_file(path)
        return policy, identity, semantics

    previous.SourceRawHistoryAdapter = MetadataCompatibleAdapter
    previous.load_cpu_actor = load
    try:
        previous.main()
    finally:
        previous.SourceRawHistoryAdapter, previous.load_cpu_actor = old_adapter, old_load


if __name__ == "__main__":
    main()
