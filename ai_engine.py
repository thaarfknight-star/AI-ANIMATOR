# -*- coding: utf-8 -*-
"""Complete on-device AI pipeline (3 stages), optimized for real hardware.

Stage 1 - Avatar generation : Stable Diffusion 1.5 img2img turns the user's
                            photo into a stylized streamer avatar.
Stage 2 - Animation keyframes: AI generates 2-3 pose keyframes per scenario;
                            procedural tweening turns them into 60 frames
                            (~10x lighter than video-diffusion, runs on iGPU).
Stage 3 - Matting           : light u2netp background removal (CPU friendly).

Optimizations: fp16 on CUDA/MPS, 24 steps @512px, models cached in an
app-local folder (downloaded once on first AI use - no pip installs),
automatic device pick: CUDA > DirectML (AMD/Intel on Windows) > MPS (macOS)
> CPU. Everything is bundled inside the exe.
"""
import os
import sys
import threading

MODEL_ID = "runwayml/stable-diffusion-v1-5"
STEPS = 24
GUIDANCE = 7.5
IMG_SIZE = 512


def app_data_dir():
    """Writable app-local folder for downloaded models (no system installs)."""
    base = os.path.dirname(os.path.abspath(__file__))
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    d = os.path.join(base, "ai_models")
    os.makedirs(d, exist_ok=True)
    return d


class NoTorchError(RuntimeError):
    pass


def pick_device():
    """Return (kind, label_fa). Never imports torch unless needed by caller."""
    try:
        import torch
    except ImportError:
        raise NoTorchError("torch داخل برنامه پیدا نشد")
    if torch.cuda.is_available():
        return "cuda", "کارت گرافیک انویدیا (CUDA)"
    try:
        import torch_directml  # noqa: F401  (Windows: AMD/Intel GPUs)
        return "dml", "کارت گرافیک (DirectML)"
    except ImportError:
        pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "گرافیک مک (MPS)"
    return "cpu", "پردازنده (کندتر، ولی کار می‌کند)"


# ------------------------------------------------------------------ prompts --
AVATAR_PROMPT = (
    "stylized digital avatar portrait of the same person, friendly streamer "
    "style, clean vector-like shading, vibrant colors, centered head and "
    "shoulders, simple gradient background, high quality"
)
AVATAR_NEG = "blurry, distorted face, extra limbs, deformed, low quality, watermark"

KEYFRAME_PROMPTS = {
    # scenario -> list of (prompt, strength)
    "working": [
        ("the same character typing on a keyboard, focused, side view", 0.55),
        ("the same character looking at a glowing monitor, front view", 0.55),
        ("the same character raising one hand with an idea, cheerful", 0.55),
    ],
    "waiting": [
        ("the same character standing idle, relaxed pose, arms crossed", 0.55),
        ("the same character tilting head slightly, patient look", 0.55),
    ],
    "create": [
        ("the same character painting on a canvas with a brush", 0.55),
        ("the same character holding up a finished colorful painting, proud", 0.55),
    ],
}


class AIEngine:
    """Owns the 3-stage pipeline. progress_cb(stage_fa, percent), log_cb(text)."""

    def __init__(self, progress_cb=None, log_cb=None):
        self.progress_cb = progress_cb or (lambda s, p: None)
        self.log_cb = log_cb or (lambda t: None)
        self._pipe = None
        self._pipe_lock = threading.Lock()
        self.device_kind = None
        self.device_label = None

    # ------------------------------------------------------------ internals --
    def _progress(self, stage, pct):
        try:
            self.progress_cb(stage, pct)
        except Exception:
            pass

    def _pipe_get(self):
        with self._pipe_lock:
            if self._pipe is not None:
                return self._pipe
            try:
                import torch
            except ImportError:
                raise NoTorchError(
                    "موتور AI داخل این نسخه نیست؛ نسخه کامل را از Artifacts دانلود کنید")
            from diffusers import StableDiffusionImg2ImgPipeline

            kind, label = pick_device()
            self.device_kind, self.device_label = kind, label
            self.log_cb(f"دستگاه محاسبه: {label}")

            use_fp16 = kind in ("cuda", "mps")
            dtype = torch.float16 if use_fp16 else torch.float32
            self.log_cb(f"دانلود/بارگذاری مدل {MODEL_ID} ... (فقط بار اول)")
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                cache_dir=app_data_dir(),
                safety_checker=None,
            )
            if kind == "cuda":
                pipe = pipe.to("cuda")
            elif kind == "mps":
                pipe = pipe.to("mps")
            elif kind == "dml":
                import torch_directml
                pipe = pipe.to(torch_directml.device())
            # cpu stays on cpu
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            self._pipe = pipe
            self.log_cb("مدل آماده شد ✅")
            return pipe

    def _img2img(self, image, prompt, strength, seed):
        try:
            import torch
        except ImportError:
            raise NoTorchError(
                "موتور AI داخل این نسخه نیست؛ نسخه کامل را از Artifacts دانلود کنید")
        pipe = self._pipe_get()
        gen = torch.Generator().manual_seed(seed)
        out = pipe(
            prompt=prompt,
            negative_prompt=AVATAR_NEG,
            image=image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
            strength=strength,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=gen,
        )
        return out.images[0]

    # -------------------------------------------------------------- stages --
    def make_avatar(self, photo):
        """Stage 1: photo -> stylized avatar (PIL RGB)."""
        self._progress("ساخت آواتار با AI", 5)
        avatar = self._img2img(photo, AVATAR_PROMPT, strength=0.45, seed=7)
        self._progress("ساخت آواتار با AI", 100)
        return avatar

    def make_keyframes(self, avatar, scenario):
        """Stage 2: avatar -> AI pose keyframes (list of PIL RGB)."""
        prompts = KEYFRAME_PROMPTS[scenario]
        frames = []
        for idx, (prompt, strength) in enumerate(prompts):
            self._progress(f"ساخت ژست {idx + 1}/{len(prompts)} ({scenario})",
                           int(100 * idx / len(prompts)))
            frames.append(self._img2img(avatar, prompt, strength, seed=100 + idx))
        self._progress("ساخت ژست‌ها", 100)
        return frames

    @staticmethod
    def remove_background(pil_img):
        """Stage 3: PIL RGB/RGBA -> RGBA with background removed.

        Uses rembg (u2netp, ~5MB, CPU friendly) when available,
        otherwise falls back to a soft circular mask.
        """
        try:
            from rembg import remove, new_session
            session = new_session("u2netp")
            return remove(pil_img.convert("RGB"), session=session)
        except Exception:
            from character import make_sprite
            return make_sprite(pil_img.convert("RGBA"), 384)

    # ------------------------------------------------------------- pipeline --
    def build_character(self, photo):
        """Run all 3 stages. Returns dict(avatar_rgba, keyframes={sc: [rgba]})."""
        self._progress("شروع", 0)
        avatar = self.make_avatar(photo)
        self._progress("حذف پس‌زمینه آواتار", 0)
        avatar_rgba = self.remove_background(avatar)

        keyframes = {}
        for sc in ("working", "waiting", "create"):
            kfs = self.make_keyframes(avatar, sc)
            keyframes[sc] = [self.remove_background(k) for k in kfs]
        self._progress("تمام شد ✅", 100)
        return {"avatar_rgba": avatar_rgba, "keyframes": keyframes}
