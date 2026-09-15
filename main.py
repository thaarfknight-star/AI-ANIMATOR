# -*- coding: utf-8 -*-
"""AI Animator — demo desktop app (PySide6).

User loads a photo -> picks a scenario (working / waiting / create-image)
-> live preview -> export transparent GIF.
"""
import os
import sys

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QRadioButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QButtonGroup, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

from PIL import Image

from character import load_photo, make_sprite, make_painting
from scenarios import render_all, SCENARIO_NAMES_FA, FPS
from exporter import export_gif


def resource_path(rel):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


ORDER = ["working", "waiting", "create"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("استودیو انیمیشن AI — دمو")
        self.setLayoutDirection(Qt.RightToLeft)
        self.resize(900, 640)

        self.photo = None
        self.sprite = None
        self.painting = None
        self.frames = []
        self.frame_idx = 0
        self.scenario = "working"

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
        lay.setSpacing(12)
        lay.setContentsMargins(18, 18, 18, 18)

        title = QLabel("🎬 استودیو انیمیشن")
        title.setStyleSheet("font-size:20px; font-weight:bold; color:#fff;")
        lay.addWidget(title)

        self.btn_load = QPushButton("📷 انتخاب عکس")
        self.btn_load.clicked.connect(self.choose_photo)
        lay.addWidget(self.btn_load)

        self.lbl_photo = QLabel("عکس نمونه فعال است")
        self.lbl_photo.setStyleSheet("color:#999; font-size:12px;")
        self.lbl_photo.setWordWrap(True)
        lay.addWidget(self.lbl_photo)

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
                "QPushButton:hover { background:#1493e0; }"
            )

        # ---- playback timer ----
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(int(1000 / FPS))

        # default sample image so the demo runs out of the box
        self.load_image(resource_path(os.path.join("assets", "sample.png")),
                        label="عکس نمونه فعال است")

    # ------------------------------------------------------------
    def choose_photo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "انتخاب عکس", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if path:
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

    def render(self):
        if self.sprite is None:
            return
        self.status.setText("در حال ساخت انیمیشن...")
        QApplication.processEvents()
        self.frames = render_all(self.sprite, self.painting, self.scenario)
        self.frame_idx = 0
        self.status.setText("آماده ✅")

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
            self, "ذخیره گیف", f"anim_{self.scenario}.gif",
            "GIF (*.gif)")
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
