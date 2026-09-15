# -*- coding: utf-8 -*-
"""AI Animator - desktop app (PySide6).

Two engines:
  * Procedural (built-in, instant): code-drawn animation from the photo.
  * AI (on-device): 3-stage pipeline - avatar generation -> pose keyframes
    -> background removal. Needs the full build (torch bundled) and downloads
    the diffusion model once on first use. Falls back to procedural on error.
"""
import os
import sys
import traceback

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QRadioButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QButtonGroup, QFrame, QSizePolicy,
    QProgressBar,
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap

from PIL import Image

from character import load_photo, make_sprite, make_painting
from scenarios import render_all, render_keyframed, SCENARIO_NAMES_FA, FPS
from exporter import export_gif


def resource_path(rel):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


ORDER = ["working", "waiting", "create"]


class AIWorker(QThread):
    progress = Signal(str, int)   # stage_fa, percent
    log = Signal(str)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, photo):
        super().__init__()
        self.photo = photo

    def run(self):
        try:
            from ai_engine import AIEngine
            engine = AIEngine(
                progress_cb=lambda s, p: self.progress.emit(s, p),
                log_cb=lambda t: self.log.emit(t),
            )
            result = engine.build_character(self.photo)
            self.done.emit(result)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("استودیو انیمیشن AI")
        self.setLayoutDirection(Qt.RightToLeft)
        self.resize(920, 660)

        self.photo = None
        self.sprite = None
        self.painting = None
        self.frames = []
        self.frame_idx = 0
        self.scenario = "working"
        self.ai_character = None   # dict from AIEngine.build_character
        self.ai_mode = False
        self.worker = None

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
        lay.setSpacing(10)
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
        self.btn_ai.clicked.connect(self.start_ai)
        lay.addWidget(self.btn_ai)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
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
        for key in ORDER:
            rb = QRadioButton(SCENARIO_NAMES_FA[key])
            rb.setStyleSheet("color:#eee; font-size:15px; padding:6px;")
            if key == self.scenario:
                rb.setChecked(True)
            rb.toggled.connect(lambda on, k=key: on and self.set_scenario(k))
            self.group.addButton(rb)
            lay.addWidget(rb)

        self.btn_export = QPushButton("💾 ذخیره گیف شفاف")
        self.btn_export.clicked.connect(self.save_gif)
        lay.addWidget(self.btn_export)

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

        # ---- playback timer ----
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(int(1000 / FPS))

        self.load_image(resource_path(os.path.join("assets", "sample.png")),
                        label="عکس نمونه فعال است")

    # ------------------------------------------------------------
    def choose_photo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "انتخاب عکس", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if path:
            self.ai_character = None
            self.ai_mode = False
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

    # ------------------------------------------------------------ AI --
    def start_ai(self):
        if self.photo is None or self.worker is not None:
            return
        try:
            from ai_engine import pick_device  # noqa
        except Exception as e:  # noqa: BLE001
            self.status.setText(f"موتور AI در دسترس نیست: {e}")
            return
        self.btn_ai.setEnabled(False)
        self.progress.show()
        self.progress.setValue(0)
        self.lbl_stage.setText("در حال شروع...")
        self.status.setText("AI در حال کار است، لطفاً صبر کنید...")
        self.worker = AIWorker(self.photo.copy())
        self.worker.progress.connect(self.on_ai_progress)
        self.worker.log.connect(self.on_ai_log)
        self.worker.done.connect(self.on_ai_done)
        self.worker.failed.connect(self.on_ai_failed)
        self.worker.start()

    def on_ai_progress(self, stage, pct):
        self.lbl_stage.setText(stage)
        self.progress.setValue(max(0, min(100, pct)))

    def on_ai_log(self, text):
        self.status.setText(text)

    def on_ai_done(self, result):
        self.worker = None
        self.ai_character = result
        self.ai_mode = True
        self.btn_ai.setEnabled(True)
        self.progress.hide()
        self.lbl_stage.setText("")
        self.status.setText("آواتار AI آماده شد ✅")
        self.render()

    def on_ai_failed(self, err):
        self.worker = None
        self.btn_ai.setEnabled(True)
        self.progress.hide()
        self.lbl_stage.setText("")
        self.status.setText(f"AI ناموفق بود، حالت نمایشی فعال است:\n{err[:160]}")

    # ------------------------------------------------------------ render --
    def render(self):
        if self.sprite is None:
            return
        self.status.setText("در حال ساخت انیمیشن...")
        QApplication.processEvents()
        if self.ai_mode and self.ai_character:
            kfs = self.ai_character["keyframes"].get(self.scenario)
            if kfs:
                self.frames = render_keyframed(kfs)
            else:
                self.frames = render_all(self.sprite, self.painting, self.scenario)
        else:
            self.frames = render_all(self.sprite, self.painting, self.scenario)
        self.frame_idx = 0
        mode = "AI 🤖" if self.ai_mode else "نمایشی"
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
