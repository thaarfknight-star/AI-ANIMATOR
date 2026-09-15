# -*- coding: utf-8 -*-
"""Complete on-device AI pipeline (3 stages) + prompt-driven scenarios.

Stage 1 - Avatar generation : Stable Diffusion 1.5 img2img turns the user's
                            photo into a stylized avatar (identity kept).
Stage 2 - Animation keyframes: AI draws 2+ pose keyframes per scenario from
                            prompts; procedural tweening turns them into a
                            smooth 60-frame loop (~10x lighter than
                            video-diffusion, runs on iGPU via DirectML/MPS).
Stage 3 - Matting           : light u2netp background removal (CPU friendly).

Scenarios are DATA (scenarios.json next to the app): the user can add new
ones from any prompt, so the program grows into a general animation tool.

Optimizations: fp16 on CUDA/MPS, 24 steps @512px, models cached in an
app-local folder (downloaded once on first AI use - no installs),
automatic device pick: CUDA > DirectML (AMD/Intel on Windows) > MPS (macOS)
> CPU. Everything is bundled inside the exe.
"""
import json
import os
import sys
import threading

MODEL_ID = "runwayml/stable-diffusion-v1-5"
STEPS = 24
GUIDANCE = 7.5
IMG_SIZE = 512
SCENARIO_CONFIG_VERSION = 1

AVATAR_PROMPT = (
    "stylized digital avatar portrait of the same person, friendly streamer "
    "style, clean vector-like shading, vibrant colors, centered head and "
    "shoulders, simple gradient background, high quality"
)
AVATAR_NEG = "blurry, distorted face, extra limbs, deformed, low quality, watermark"

DEFAULT_SCENARIOS = [
    {
        "id": "working",
        "name_fa": "در حال کار",
        "strength": 0.55,
        "prompts": [
            "the same character typing on a keyboard, focused, side view",
            "the same character looking at a glowing monitor, front view",
            "the same character raising one hand with an idea, cheerful",
        ],
    },
    {
        "id": "waiting",
        "name_fa": "در انتظار",
        "strength": 0.55,
        "prompts": [
            "the same character standing idle, relaxed pose, arms crossed",
            "the same character tilting head slightly, patient look",
        ],
    },
    {
        "id": "create",
        "name_fa": "ساخت عکس",
        "strength": 0.55,
        "prompts": [
            "the same character painting on a canvas with a brush",
            "the same character holding up a finished colorful painting, proud",
        ],
    },
]


def app_data_dir():
    """Writable app-local folder for models + scenario config (no installs)."""
    base = os.path.dirname(os.path.abspath(__file__))
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    d = os.path.join(base, "ai_models")
    os.makedirs(d, exist_ok=True)
    return d


def scenarios_path():
    return os.path.join(app_data_dir(), "scenarios.json")


def load_scenarios():
    """Load user scenarios; seed with defaults on first run; merge new ids."""
    path = scenarios_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            scenarios = data.get("scenarios", [])
            have = {s["id"] for s in scenarios}
            # merge in any new built-in scenarios from updates
            for default in DEFAULT_SCENARIOS:
                if default["id"] not in have:
                    scenarios.insert(0, dict(default))
            return scenarios
        except Exception:
            pass
    scenarios = [dict(s) for s in DEFAULT_SCENARIOS]
    save_scenarios(scenarios)
    return scenarios


def save_scenarios(scenarios):
    with open(scenarios_path(), "w", encoding="utf-8") as f:
        json.dump({"version": SCENARIO_CONFIG_VERSION,
                   "scenarios": scenarios}, f, ensure_ascii=False, indent=2)


class NoTorchError(RuntimeError):
    pass


def pick_device():
    """Return (kind, label_fa). Raises NoTorchError when torch is missing."""
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


def check_engine():
    """Self-test: import every AI component. Returns [(name_fa, ok, detail)].

    Catches packaging bugs (like the diffusers/huggingface_hub import break)
    before the heavy model download starts.
    """
    out = []

    def _ver(mod):
        return getattr(mod, "__version__", "?")

    try:
        import torch
        out.append(("torch", True, _ver(torch)))
    except Exception as e:  # noqa: BLE001
        out.append(("torch", False, str(e)[:160]))
        return out
    try:
        import torch_directml  # noqa: F401
        try:
            from importlib.metadata import version as _pkg_version
            _dml_ver = _pkg_version("torch-directml")
        except Exception:
            _dml_ver = _ver(sys.modules["torch_directml"])
        out.append(("torch-directml", True, _dml_ver))
    except Exception:  # noqa: BLE001
        out.append(("torch-directml", False, "موجود نیست (فقط ویندوز/اختیاری)"))
    try:
        # the exact import that broke on the built exe before pinning
        from diffusers import StableDiffusionImg2ImgPipeline  # noqa: F401
        import diffusers
        out.append(("diffusers", True, _ver(diffusers)))
    except Exception as e:  # noqa: BLE001
        out.append(("diffusers", False, str(e)[:200]))
    try:
        import transformers
        out.append(("transformers", True, _ver(transformers)))
    except Exception as e:  # noqa: BLE001
        out.append(("transformers", False, str(e)[:160]))
    try:
        import huggingface_hub
        # diffusers<0.31 needs cached_download, removed in huggingface_hub>=0.26
        ok = hasattr(huggingface_hub, "cached_download")
        out.append(("huggingface_hub", True,
                    _ver(huggingface_hub) + (" (cached_download ✅)" if ok else " (cached_download ❌)")))
    except Exception as e:  # noqa: BLE001
        out.append(("huggingface_hub", False, str(e)[:160]))
    try:
        from rembg import new_session  # noqa: F401
        import rembg
        out.append(("rembg", True, _ver(rembg)))
    except Exception as e:  # noqa: BLE001
        out.append(("rembg", False, str(e)[:160]))
    try:
        kind, label = pick_device()
        out.append(("دستگاه محاسبه", True, f"{label} [{kind}]"))
    except Exception as e:  # noqa: BLE001
        out.append(("دستگاه محاسبه", False, str(e)[:160]))
    return out


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

    def _from_pretrained_with_fallback(self, dtype):
        """بارگذاری مدل با تلاش دوباره از آینه‌ی hf-mirror.com.

        در بعضی شبکه‌ها (از جمله ایران) API سایت huggingface.co جواب
        می‌دهد ولی CDN دانلود فایل (us.aws.cdn.hf.co) مسدود است و دانلود
        با MaxRetryError می‌افتد. در این حالت یک بار دیگر از آینه تلاش
        می‌کنیم وگرنه خطای فارسی و واضح بالا می‌بریم.
        """
        import os
        from diffusers import StableDiffusionImg2ImgPipeline

        last_err = None
        for endpoint in (None, "https://hf-mirror.com"):
            if endpoint:
                os.environ["HF_ENDPOINT"] = endpoint
                try:
                    import huggingface_hub.constants as _c
                    _c.HF_ENDPOINT = endpoint
                except Exception:
                    pass
                self.log_cb("سرور اصلی جواب نداد؛ تلاش دوباره با آینه‌ی "
                            "hf-mirror.com ...")
            try:
                return StableDiffusionImg2ImgPipeline.from_pretrained(
                    MODEL_ID,
                    torch_dtype=dtype,
                    cache_dir=app_data_dir(),
                    safety_checker=None,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                where = "سرور اصلی" if endpoint is None else "آینه"
                self.log_cb(f"❌ دانلود از {where} ناموفق بود: "
                            f"{str(e)[:220]}")
        raise RuntimeError(
            "دانلود مدل ناموفق بود. اینترنت را بررسی کنید؛ اگر CDN سایت "
            "HuggingFace در شبکه‌ی شما مسدود است، با VPN یک بار مدل را "
            "دانلود کنید (فقط بار اول لازم است). "
            f"جزئیات: {str(last_err)[:300]}")

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
            pipe = self._from_pretrained_with_fallback(dtype)
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

    def make_keyframes(self, avatar, prompts, strength, tag):
        """Stage 2: avatar + prompts -> AI pose keyframes (list of PIL RGB)."""
        frames = []
        for idx, prompt in enumerate(prompts):
            self._progress(f"ساخت ژست {idx + 1}/{len(prompts)} ({tag})",
                           int(100 * idx / len(prompts)))
            frames.append(self._img2img(avatar, prompt, strength,
                                        seed=1000 + idx))
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
    def build_character(self, photo, scenarios):
        """Run all 3 stages. Returns dict(avatar_rgb, avatar_rgba,
        keyframes={scenario_id: [rgba]})."""
        self._progress("شروع", 0)
        avatar = self.make_avatar(photo)
        self._progress("حذف پس‌زمینه آواتار", 0)
        avatar_rgba = self.remove_background(avatar)

        keyframes = {}
        for sc in scenarios:
            kfs = self.make_keyframes(avatar, sc["prompts"],
                                      sc.get("strength", 0.55), sc["name_fa"])
            keyframes[sc["id"]] = [self.remove_background(k) for k in kfs]
        self._progress("تمام شد ✅", 100)
        return {"avatar_rgb": avatar, "avatar_rgba": avatar_rgba,
                "keyframes": keyframes}
