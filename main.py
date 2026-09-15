# -*- coding: utf-8 -*-
"""AI Animator - desktop app (PySide6).

Two engines:
  * Procedural (built-in, instant): code-drawn animation from the photo.
  * AI (on-device): 3-stage pipeline - avatar generation -> pose keyframes
    -> background removal. Needs the full build (torch bundled) and downloads
    the diffusion model once on first use. Falls back to procedural on error.

Scenarios are user-extensible: "new scenario from prompt" turns any list of
prompts into a permanent, re-renderable animation scenario.
"""
import os
import sys
import traceback

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QRadioButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QButtonGroup, QFrame, QSizePolicy,
    QProgressBar, QTextEdit, QDialog, QLineEdit, QPlainTextEdit,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap

from PIL import Image

from character import load_photo, make_sprite, make_painting
from scenarios import render_all, render_keyframed, FPS
from exporter import export_gif


def resource_path(rel):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


BUILTIN_IDS = ("working", "waiting", "create")


class AIWorker(QThread):
    progress = Signal(str, int)   # stage_fa, percent
    log = Signal(str)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, job, photo, avatar_rgb=None):
        super().__init__()
        self.job = job            # {"type": "full"} | {"type": "scenario", "scenario": {...}}
        self.photo = photo
        self.avatar_rgb = avatar_rgb  # reuse avatar across scenario jobs

    def run(self):
        try:
            from ai_engine import AIEngine, load_scenarios
            engine = AIEngine(
                progress_cb=lambda s, p: self.progress.emit(s, p),
                log_cb=lambda t: self.log.emit(t),
            )
            if self.job["type"] == "full":
                result = engine.build_character(self.photo, load_scenarios())
            else:
                sc = self.job["scenario"]
                avatar = self.avatar_rgb
                avatar_rgba = None
                if avatar is None:
                    self.log.emit("ساخت آواتار (برای سناریوی جدید)...")
                    avatar = engine.make_avatar(self.photo)
                    avatar_rgba = engine.remove_background(avatar)
                kfs = engine.make_keyframes(avatar, sc["prompts"],
                                            sc.get("strength", 0.55),
                                            sc["name_fa"])
                kf_rgba = [engine.remove_background(k) for k in kfs]
                result = {"avatar_rgb": avatar, "avatar_rgba": avatar_rgba,
                          "keyframes": {sc["id"]: kf_rgba}, "scenario": sc}
            self.done.emit(result)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(str(e)[:400])


class CheckWorker(QThread):
    finished = Signal(list)

    def run(self):
        try:
            from ai_engine import check_engine
            self.finished.emit(check_engine())
        except Exception as e:  # noqa: BLE001
            self.finished.emit([("بررسی", False, str(e)[:200])])


class ScenarioDialog(QDialog):
    """New scenario from prompts: name (Persian) + one English prompt per line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("سناریوی جدید از پرامپت")
        self.setLayoutDirection(Qt.RightToLeft)
        self.resize(440, 380)
        lay = QVBoxLayout(self)

        lay.addWidget(QLabel("نام سناریو (فارسی):"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("مثلاً: در حال بازی")
        lay.addWidget(self.name_edit)

        lay.addWidget(QLabel("پرامپت‌ها (انگلیسی، هر خط یک ژست):"))
        self.prompts_edit = QPlainTextEdit()
        self.prompts_edit.setPlaceholderText(
            "the same character playing guitar on stage\n"
            "the same character jumping with joy, confetti")
        lay.addWidget(self.prompts_edit)

        hint = QLabel("نکته: هر پرامپت با «the same character» شروع شود تا کاراکتر ثابت بماند.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:11px;")
        lay.addWidget(hint)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.button(QDialogButtonBox.Ok).setText("ساخت انیمیشن")
        box.button(QDialogButtonBox.Cancel).setText("انصراف")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

    def values(self):
        name = self.name_edit.text().strip()
        prompts = [l.strip() for l in self.prompts_edit.toPlainText().splitlines()
                   if l.strip()]
        return name, prompts


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("استودیو انیمیشن AI")
        self.setLayoutDirection(Qt.RightToLeft)
        self.resize(960, 700)

        from ai_engine import load_scenarios
        self.scenarios = load_scenarios()

        self.photo = None
        self.sprite = None
        self.painting = None
        self.frames = []
        self.frame_idx = 0
        self.scenario = self.scenarios[0]["id"]
        self.ai_keyframes = {}     # scenario_id -> [RGBA]
        self.ai_avatar_rgb = None  # reused for new scenarios/prompts
        self.worker = None
        self.check_worker = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ---- preview ----
        self.preview = QLabel("عکس را انتخاب کنید")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(520, 520)
        self.preview.setStyleSheet(
            "QLabel { background:#14141c; border-radius:12px; color:#888; font-size:16px; }"
        )
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.preview, stretch=3)

        # ---- side panel ----
        panel = QFrame()
        panel.setStyleSheet("QFrame { background:#1c1c26; border-radius:12px; }")
        lay = QVBoxLayout(panel)
        lay.setSpacing(8)
        lay.setContentsMargins(18, 18, 18, 18)

        title = QLabel("🎬 استودیو انیمیشن AI")
        title.setStyleSheet("font-size:20px; font-weight:bold; color:#fff;")
        lay.addWidget(title)

        self.btn_load = QPushButton("📷 انتخاب عکس")
        self.btn_load.clicked.connect(self.choose_photo)
        lay.addWidget(self.btn_load)

        self.lbl_photo = QLabel("عکس نمونه فعال است")
        self.lbl_photo.setStyleSheet("color:#999; font-size:12px;")
        self.lbl_photo.setWordWrap(True)
        lay.addWidget(self.lbl_photo)

        self.btn_ai = QPushButton("🤖 ساخت آواتار با AI")
        self.btn_ai.clicked.connect(self.start_ai_full)
        lay.addWidget(self.btn_ai)

        self.btn_prompt = QPushButton("✨ سناریوی جدید از پرامپت")
        self.btn_prompt.clicked.connect(self.new_scenario_dialog)
        lay.addWidget(self.btn_prompt)

        self.btn_check = QPushButton("🔍 بررسی موتور AI")
        self.btn_check.clicked.connect(self.run_check)
        lay.addWidget(self.btn_check)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setStyleSheet(
            "QProgressBar { background:#101018; border-radius:6px; color:#ddd; }"
            "QProgressBar::chunk { background:#0f7cc1; border-radius:6px; }")
        self.progress.hide()
        lay.addWidget(self.progress)

        self.lbl_stage = QLabel("")
        self.lbl_stage.setStyleSheet("color:#9db4d0; font-size:12px;")
        self.lbl_stage.setWordWrap(True)
        lay.addWidget(self.lbl_stage)

        lay.addWidget(QLabel("سناریو:"))
        self.group = QButtonGroup(self)
        self.radio_box = QVBoxLayout()
        self.radio_box.setSpacing(2)
        lay.addLayout(self.radio_box)
        self.rebuild_radios()

        self.btn_export = QPushButton("💾 ذخیره گیف شفاف")
        self.btn_export.clicked.connect(self.save_gif)
        lay.addWidget(self.btn_export)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(110)
        self.log_view.setStyleSheet(
            "QTextEdit { background:#101018; color:#9db4d0; font-size:11px;"
            " border-radius:6px; }")
        self.log_view.setPlaceholderText("گزارش موتور AI اینجا نمایش داده می‌شود…")
        lay.addWidget(self.log_view)

        lay.addStretch(1)
        self.status = QLabel("")
        self.status.setStyleSheet("color:#8f8; font-size:13px;")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        hint = QLabel("خروجی: گیف ۵۱۲×۵۱۲ با پس‌زمینه شفاف، مناسب اورلی استریم")
        hint.setStyleSheet("color:#777; font-size:11px;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        root.addWidget(panel, stretch=1)

        for b in (self.btn_load, self.btn_export):
            b.setStyleSheet(
                "QPushButton { background:#0f7cc1; color:#fff; border-radius:8px;"
                " padding:10px; font-size:14px; }"
                "QPushButton:hover { background:#1493e0; }")
        self.btn_ai.setStyleSheet(
            "QPushButton { background:#7b2fbe; color:#fff; border-radius:8px;"
            " padding:10px; font-size:14px; }"
            "QPushButton:hover { background:#9440dd; }"
            "QPushButton:disabled { background:#4a4a55; color:#999; }")
        for b in (self.btn_prompt, self.btn_check):
            b.setStyleSheet(
                "QPushButton { background:#2a2a3a; color:#ddd; border-radius:8px;"
                " padding:9px; font-size:13px; }"
                "QPushButton:hover { background:#3a3a4e; }"
                "QPushButton:disabled { background:#22222c; color:#777; }")

        # ---- playback timer ----
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(int(1000 / FPS))

        self.load_image(resource_path(os.path.join("assets", "sample.png")),
                        label="عکس نمونه فعال است")

    # ------------------------------------------------------------ scenarios --
    def rebuild_radios(self):
        while self.radio_box.count():
            item = self.radio_box.takeAt(0)
            w = item.widget()
            if w:
                self.group.removeButton(w)
                w.deleteLater()
        for sc in self.scenarios:
            rb = QRadioButton(sc["name_fa"])
            rb.setStyleSheet("color:#eee; font-size:15px; padding:5px;")
            if sc["id"] == self.scenario:
                rb.setChecked(True)
            rb.toggled.connect(lambda on, k=sc["id"]: on and self.set_scenario(k))
            self.group.addButton(rb)
            self.radio_box.addWidget(rb)

    # ------------------------------------------------------------
    def choose_photo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "انتخاب عکس", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if path:
            self.ai_keyframes = {}
            self.ai_avatar_rgb = None
            self.load_image(path, label=os.path.basename(path))

    def load_image(self, path, label=""):
        try:
            self.photo = load_photo(path)
        except Exception as e:  # noqa: BLE001
            self.status.setText(f"خطا در باز کردن عکس: {e}")
            return
        self.status.setText("در حال آماده‌سازی...")
        QApplication.processEvents()
        self.sprite = make_sprite(self.photo, 300)
        self.painting = make_painting(self.photo)
        self.lbl_photo.setText(label)
        self.render()

    def set_scenario(self, key):
        self.scenario = key
        self.render()

    def log(self, text):
        self.log_view.append(text)

    # ------------------------------------------------------------ AI --
    def start_ai_full(self):
        if self.photo is None or self.worker is not None:
            return
        self._busy(True, "AI در حال کار است، لطفاً صبر کنید...")
        self.worker = AIWorker({"type": "full"}, self.photo.copy())
        self._wire_worker(self.worker)
        self.worker.start()

    def new_scenario_dialog(self):
        if self.photo is None or self.worker is not None:
            return
        dlg = ScenarioDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        name, prompts = dlg.values()
        if not name or not prompts:
            self.status.setText("نام و حداقل یک پرامپت لازم است")
            return
        import time
        from ai_engine import save_scenarios
        sc = {"id": f"custom_{int(time.time())}", "name_fa": name,
              "strength": 0.55, "prompts": prompts}
        self.scenarios.append(sc)
        save_scenarios(self.scenarios)
        self.rebuild_radios()
        self.set_scenario(sc["id"])
        self._busy(True, f"در حال ساخت سناریوی «{name}»...")
        self.log(f"✨ سناریوی جدید: {name} ({len(prompts)} ژست)")
        self.worker = AIWorker({"type": "scenario", "scenario": sc},
                               self.photo.copy(), self.ai_avatar_rgb)
        self._wire_worker(self.worker)
        self.worker.start()

    def run_check(self):
        if self.check_worker is not None:
            return
        self.btn_check.setEnabled(False)
        self.log("🔍 بررسی موتور AI...")
        self.check_worker = CheckWorker()
        self.check_worker.finished.connect(self.on_check_done)
        self.check_worker.start()

    def on_check_done(self, results):
        self.check_worker = None
        self.btn_check.setEnabled(True)
        for name, ok, detail in results:
            self.log(("✅ " if ok else "❌ ") + f"{name}: {detail}")
        bad = [n for n, ok, _ in results if not ok and n != "torch-directml"]
        if bad:
            self.status.setText("موتور AI مشکل دارد؛ گزارش بالا را بفرستید")
        else:
            self.status.setText("موتور AI سالم است ✅")

    def _busy(self, busy, msg=""):
        for b in (self.btn_ai, self.btn_prompt, self.btn_check, self.btn_load):
            b.setEnabled(not busy)
        if busy:
            self.progress.show()
            self.progress.setValue(0)
            self.lbl_stage.setText("در حال شروع...")
            self.status.setText(msg)
        else:
            self.progress.hide()
            self.lbl_stage.setText("")

    def _wire_worker(self, worker):
        worker.progress.connect(self.on_ai_progress)
        worker.log.connect(self.log)
        worker.done.connect(self.on_ai_done)
        worker.failed.connect(self.on_ai_failed)

    def on_ai_progress(self, stage, pct):
        self.lbl_stage.setText(stage)
        self.progress.setValue(max(0, min(100, pct)))

    def on_ai_done(self, result):
        self.worker = None
        if result.get("avatar_rgb") is not None:
            self.ai_avatar_rgb = result["avatar_rgb"]
        for sid, kfs in result["keyframes"].items():
            self.ai_keyframes[sid] = kfs
        if result.get("scenario"):
            self.set_scenario(result["scenario"]["id"])
        self._busy(False)
        self.status.setText("آواتار AI آماده شد ✅")
        self.log("تمام شد ✅")
        self.render()

    def on_ai_failed(self, err):
        self.worker = None
        self._busy(False)
        self.status.setText("AI ناموفق بود، حالت نمایشی فعال است")
        self.log(f"❌ خطا: {err}")
        self.log("💡 با دکمه «بررسی موتور AI» جزئیات را ببینید")

    # ------------------------------------------------------------ render --
    def render(self):
        if self.sprite is None:
            return
        self.status.setText("در حال ساخت انیمیشن...")
        QApplication.processEvents()
        kfs = self.ai_keyframes.get(self.scenario)
        if kfs:
            self.frames = render_keyframed(kfs)
            mode = "AI 🤖"
        elif self.scenario in BUILTIN_IDS:
            self.frames = render_all(self.sprite, self.painting, self.scenario)
            mode = "نمایشی"
        else:
            self.frames = render_all(self.sprite, self.painting, "waiting")
            mode = "نمایشی (اول AI را اجرا کنید)"
        self.frame_idx = 0
        self.status.setText(f"آماده ✅ (حالت {mode})")

    def next_frame(self):
        if not self.frames:
            return
        pil = self.frames[self.frame_idx]
        self.frame_idx = (self.frame_idx + 1) % len(self.frames)
        data = pil.tobytes("raw", "RGBA")
        qimg = QImage(data, pil.width, pil.height, QImage.Format_RGBA8888)
        pm = QPixmap.fromImage(qimg).scaled(
            512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pm)

    def save_gif(self):
        if not self.frames:
            self.status.setText("اول انیمیشن را بسازید")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "ذخیره گیف", f"anim_{self.scenario}.gif", "GIF (*.gif)")
        if not path:
            return
        self.status.setText("در حال ذخیره...")
        QApplication.processEvents()
        try:
            export_gif(self.frames, path, fps=FPS)
            self.status.setText(f"ذخیره شد ✅\n{path}")
        except Exception as e:  # noqa: BLE001
            self.status.setText(f"خطا در ذخیره: {e}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
