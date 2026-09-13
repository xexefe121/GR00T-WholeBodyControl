"""MJLab configuration shim; no architecture or behavior change from compact v1."""

from gear_sonic.trl.mjlab.native23_compact_actor import CompactNative23Actor


class CompactNative23ActorV2(CompactNative23Actor):
    def __init__(self, *args, cnn_cfg=None, **kwargs):
        if cnn_cfg is not None:
            raise ValueError("compact tracker does not accept a CNN")
        super().__init__(*args, **kwargs)
