# -*- coding: utf-8 -*-
"""Transparent GIF export (PIL only, no ffmpeg needed)."""
from PIL import Image


def _rgba_to_gif_frame(img):
    """Convert RGBA frame to P-mode with index 255 as transparency."""
    alpha = img.getchannel("A")
    pal = img.convert("RGB").quantize(colors=255, method=Image.MEDIANCUT)
    transparent = alpha.point(lambda v: 255 if v < 128 else 0)
    pal.paste(255, transparent)
    return pal


def export_gif(frames, path, fps=15):
    """Export a list of RGBA PIL frames to a transparent looping GIF."""
    if not frames:
        raise ValueError("no frames to export")
    pal_frames = [_rgba_to_gif_frame(f) for f in frames]
    duration_ms = int(1000 / fps)
    pal_frames[0].save(
        path,
        save_all=True,
        append_images=pal_frames[1:],
        duration=duration_ms,
        loop=0,
        transparency=255,
        disposal=2,
    )
    return path
