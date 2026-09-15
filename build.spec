# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for AI Animator (Windows + macOS).

Bundles the full on-device AI stack (torch, diffusers, rembg, ...) so the
app runs with zero installs on the user's PC. Diffusion model weights are
NOT bundled - they download once into ai_models/ on first AI use.
"""
from PyInstaller.utils.hooks import collect_all, copy_metadata

datas = [("assets", "assets")]
binaries = []
hiddenimports = []

# Collect full package content (code + data files) for the whole AI stack.
for pkg in ("torch", "diffusers", "transformers", "tokenizers",
            "huggingface_hub", "safetensors", "rembg", "onnxruntime",
            "PIL", "numpy", "requests", "urllib3", "certifi",
            "charset_normalizer", "idna", "tqdm", "packaging",
            "filelock", "fsspec", "regex"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# Package *metadata* (dist-info): libraries like huggingface_hub/transformers
# call importlib.metadata at runtime and crash without it ("No package
# metadata was found for ..."). This was the missing piece.
for pkg in ("requests", "urllib3", "certifi", "transformers", "tokenizers",
            "diffusers", "huggingface_hub", "safetensors", "filelock",
            "fsspec", "tqdm", "packaging", "regex", "numpy", "Pillow",
            "torch", "onnxruntime", "rembg"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# Windows-only: AMD/Intel GPU backend (absent on macOS - skipped there)
try:
    d, b, h = collect_all("torch_directml")
    datas += d
    binaries += b
    hiddenimports += h
except Exception:
    pass

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "scipy", "pandas", "notebook", "jupyter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AIAnimator",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="AIAnimator",
)
