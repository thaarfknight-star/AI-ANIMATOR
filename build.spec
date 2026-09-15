# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for AI Animator (Windows + macOS).

Bundles the full on-device AI stack (torch, diffusers, rembg, ...) so the
app runs with zero installs on the user's PC. Diffusion model weights are
NOT bundled - they download once into ai_models/ on first AI use.
"""
from PyInstaller.utils.hooks import collect_all

datas = [("assets", "assets")]
binaries = []
hiddenimports = []

for pkg in ("torch", "diffusers", "huggingface_hub", "safetensors",
            "rembg", "onnxruntime", "PIL"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
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
