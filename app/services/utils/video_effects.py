import numpy as np
from moviepy import Clip, ColorClip, CompositeVideoClip, vfx
from PIL import Image


# FadeIn
def fadein_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeIn(t)])


# FadeOut
def fadeout_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeOut(t)])


# SlideIn
def slidein_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size

    # MoviePy's built-in SlideIn is unstable for full-screen material in this processing
    # chain: the transition is logically applied but the picture barely changes. Use an
    # explicit black background plus a position animation instead, so the transition is
    # visible and its behavior controllable.
    def position(current_time: float):
        progress = min(max(current_time / max(t, 0.001), 0), 1)

        if side == "left":
            return (-width + width * progress, 0)
        if side == "right":
            return (width - width * progress, 0)
        if side == "top":
            return (0, -height + height * progress)
        if side == "bottom":
            return (0, height - height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# SlideOut
def slideout_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size
    transition_duration = min(t, clip.duration)
    transition_start = max(clip.duration - transition_duration, 0)

    # SlideOut is also replaced by an explicit position animation so the clip reliably slides
    # out at its end.
    def position(current_time: float):
        if current_time <= transition_start:
            return (0, 0)

        progress = min(
            max((current_time - transition_start) / max(transition_duration, 0.001), 0), 1
        )

        if side == "left":
            return (-width * progress, 0)
        if side == "right":
            return (width * progress, 0)
        if side == "top":
            return (0, -height * progress)
        if side == "bottom":
            return (0, height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# Keep the original 20% zoom range so even ~3-second clips show a clearly visible Ken Burns
# motion. Zoom stability comes from the sub-pixel center sampling below, not from weakening
# the effect to hide flicker in the source video encoding.
_ZOOM_MAX_SCALE = 1.2


def _zoom_frame(frame: np.ndarray, scale_factor: float) -> np.ndarray:
    """Zoom without black borders and without jitter using a sub-pixel centered crop.

    The crop width/height must not be converted to integers first: as the scale changes
    continuously, integer boundaries jump in uneven steps and the half-pixel sampling phase
    changes whenever the size flips between odd and even, which shows up as jitter. Pillow's
    EXTENT transform accepts float boundaries directly and does sub-pixel sampling on a fixed
    output canvas; the left/right and top/bottom boundaries stay symmetric around the same
    float center, which suits a slow continuous zoom over a whole clip.
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor must be greater than zero")

    # A scale of 1 returns the original frame to avoid a pointless resample that slightly blurs
    # the first frame.
    if abs(scale_factor - 1.0) < 1e-9:
        return frame

    height, width = frame.shape[:2]
    crop_width = width / scale_factor
    crop_height = height / scale_factor
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = left + crop_width
    bottom = top + crop_height

    image = Image.fromarray(frame)
    transformed = image.transform(
        (width, height),
        Image.Transform.EXTENT,
        (left, top, right, bottom),
        # Continuous video zoom cares more about consistency between adjacent frames.
        # BICUBIC/LANCZOS are sharper per frame but high-frequency texture crossing the
        # sampling grid tends to ring and flicker in brightness; BILINEAR is softer and trades
        # a little sharpness for a more stable motion impression.
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(transformed)


def zoomin_transition(clip: Clip, t: float) -> Clip:
    """Zoom smoothly from the original picture to 1.2x over the whole clip."""
    # t is kept for now to share the call signature of the other transition functions; the
    # zoom must span the whole clip, otherwise the picture freezes abruptly after a short
    # zoom, which does not suit static or low-motion material.
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = 1 + (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect, apply_to=["mask"])


def zoomout_transition(clip: Clip, t: float) -> Clip:
    """Zoom smoothly from 1.2x back to the original picture over the whole clip."""
    # As in zoomin_transition, t only exists for the shared transition call interface.
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = _ZOOM_MAX_SCALE - (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect, apply_to=["mask"])
