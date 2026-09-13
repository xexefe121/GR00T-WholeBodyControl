"""Component-selective learned clipping feedback; no proposal arithmetic changes."""
import numpy as np


def clipped_component_feedback(combined, raw_position, target, contract, learned):
    """Keep the original raw float32 action except at learned target clips.

    The applied inverse uses the existing teacher/diagnostic operation order.
    Copy-and-select avoids a float64 round trip on every unclipped component.
    Native clipping is recorded independently from the enabled feedback mask.
    """
    combined=np.asarray(combined)
    raw_position=np.asarray(raw_position);target=np.asarray(target)
    if combined.shape!=(23,) or combined.dtype!=np.float32:
        raise ValueError('Original combined action must be float32[23].')
    if raw_position.shape!=(23,) or target.shape!=(23,):
        raise ValueError('Position proposal and target must be [23].')
    if not all(np.isfinite(x).all() for x in (combined,raw_position,target)):
        raise ValueError('Nonfinite proposal or action.')
    native_clip=raw_position!=target
    mask=native_clip.copy() if learned else np.zeros(23,dtype=bool)
    feedback=combined.copy()
    if np.any(mask):
        actual=((target-np.asarray(contract['default_q']))*np.asarray(contract['kp'])/
                (.25*np.asarray(contract['training_effort']))).astype(np.float32)
        if not np.isfinite(actual).all():raise ValueError('Nonfinite applied action inverse.')
        feedback[mask]=actual[mask]
    return feedback,native_clip,mask
