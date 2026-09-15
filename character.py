# -*- coding: utf-8 -*-
"""Character sprite preparation: photo -> circular avatar sprite."""
from PIL import Image, ImageDraw, ImageFilter


def load_photo(path):
    """Load any image file as RGBA."""
    return Image.open(path).convert("RGBA")


def make_sprite(photo, size=300):
    """Center-crop to square, resize, apply circular soft-edge mask."""
    w, h = photo.size
    s = min(w, h)
    img = photo.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((size, size), Image.LANCZOS)

    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    m = max(6, size // 40)
    d.ellipse((m, m, size - m, size - m), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(3, size // 50)))
    img.putalpha(mask)
    return img


def make_painting(photo, size=360, colors=14):
    """Stylized 'AI painting' derived from the input photo.

    Used by the create-image scenario so the generated artwork always
    matches the photo the user submitted.
    """
    img = photo.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((size, size), Image.LANCZOS)
    img = img.filter(ImageFilter.GaussianBlur(1.1))
    q = img.quantize(colors=colors, method=Image.MEDIANCUT).convert("RGB")
    # gentle saturation boost for a painted feel
    from PIL import ImageEnhance
    q = ImageEnhance.Color(q).enhance(1.18)
    q = ImageEnhance.Contrast(q).enhance(1.06)
    return q.convert("RGBA")
