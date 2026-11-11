import os
import sys

import requests
from PyQt5.QtCore import QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                             QPushButton, QSplitter, QTextEdit, QVBoxLayout, QWidget)

# ==================== 配置 ====================
# 后端地址，cpolar 穿透时设置环境变量 ROAD_API_URL=https://xxx.cpolar.io
API_BASE = os.environ.get("ROAD_API_URL", "http://127.0.0.1:5000").rstrip("/")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class PredictWorker(QThread):
    """后台线程请求 Flask 接口，避免推理期间界面卡死"""

    success = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path

    def run(self):
        try:
            with open(self.image_path, "rb") as f:
                resp = requests.post(
                    f"{API_BASE}/predict",
                    files={"image": (os.path.basename(self.image_path), f, "image/jpeg")},
                    timeout=120,
                )
            payload = resp.json()
            if resp.status_code != 200 or payload.get("code") != 200:
                self.failed.emit(payload.get("msg", f"接口返回异常（HTTP {resp.status_code}）"))
                return

            data = payload["data"]
            img_resp = requests.get(data["image_url"], timeout=60)
            img_resp.raise_for_status()
            data["result_bytes"] = img_resp.content
            self.success.emit(data)
        except requests.exceptions.Timeout:
            self.failed.emit("请求超时，请检查后端是否繁忙或网络是否通畅")
        except requests.exceptions.ConnectionError:
            self.failed.emit(f"无法连接后端服务：{API_BASE}\n请先运行 python app.py 启动 Flask")
        except ValueError:
            self.failed.emit("后端返回的内容不是合法 JSON")
        except Exception as err:
            self.failed.emit(f"请求异常：{err}")


class ImagePanel(QWidget):
    """保持等比缩放、居中显示的图片面板"""

    PADDING = 10

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._pixmap = None

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setObjectName("panelTitle")

        self.image_label = QLabel("暂无图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 340)
        self.image_label.setObjectName("imageBox")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, 1)

    def set_image_bytes(self, data):
        pix = QPixmap()
        if not pix.loadFromData(data):
            self.image_label.setText("结果图加载失败")
            return
        self._pixmap = pix
        self._refresh_later()

    def set_image_path(self, path):
        pix = QPixmap(path)
        if pix.isNull():
            self.image_label.setText("原图加载失败")
            return
        self._pixmap = pix
        self._refresh_later()

    def _refresh_later(self):
        # 等布局完成后再缩放，避免首次显示时尺寸未就绪
        self._refresh()
        QTimer.singleShot(0, self._refresh)

    def _refresh(self):
        if self._pixmap is None:
            return
        size = self.image_label.size()
        target = QSize(max(1, size.width() - self.PADDING * 2),
                       max(1, size.height() - self.PADDING * 2))
        self.image_label.setPixmap(
            self._pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()


class RoadDetectWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("道路缺陷与障碍物检测系统")
        self.resize(1200, 760)
        self.worker = None

        self.btn_select = QPushButton("选择道路图片检测")
        self.btn_select.setMinimumHeight(38)
        self.btn_select.clicked.connect(self.select_and_predict)

        self.status_label = QLabel("等待选择图片")
        self.status_label.setObjectName("statusLabel")

        self.panel_origin = ImagePanel("原图")
        self.panel_result = ImagePanel("检测结果")

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("检测统计结果将在这里显示")
        self.result_text.setMinimumWidth(280)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.panel_origin)
        splitter.addWidget(self.panel_result)
        splitter.addWidget(self.result_text)
        splitter.setSizes([420, 420, 320])

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.btn_select)
        top_bar.addWidget(self.status_label, 1)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addLayout(top_bar)
        layout.addWidget(splitter, 1)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self._apply_style()
        self._check_server()

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #f5f6f8; }
            QPushButton {
                background: #2f6fed; color: white; border: none;
                border-radius: 6px; padding: 8px 18px; font-size: 15px;
            }
            QPushButton:hover { background: #1f5cd6; }
            QPushButton:disabled { background: #a9bbdd; }
            QLabel#panelTitle { color: #333; font-size: 14px; font-weight: bold; padding: 4px; }
            QLabel#imageBox {
                background: #ffffff; border: 1px solid #dcdfe6;
                border-radius: 8px; color: #999;
            }
            QLabel#statusLabel { color: #666; font-size: 13px; padding-left: 6px; }
            QTextEdit {
                background: #ffffff; border: 1px solid #dcdfe6;
                border-radius: 8px; padding: 8px; font-size: 14px;
            }
        """)

    def _check_server(self):
        """启动时探测一次后端，方便定位“连不上”的问题"""
        try:
            resp = requests.get(f"{API_BASE}/health", timeout=3)
            if resp.status_code == 200:
                self.status_label.setText("后端服务已连接")
                return
        except Exception:
            pass
        self.status_label.setText(f"后端未连接：{API_BASE}，请先启动 app.py")

    def select_and_predict(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not file_path:
            return

        self.panel_origin.set_image_path(file_path)
        self.panel_result.image_label.setText("检测中...")
        self.panel_result.image_label.setPixmap(QPixmap())
        self.result_text.clear()
        self.btn_select.setEnabled(False)
        self.status_label.setText("正在上传并推理...")

        self.worker = PredictWorker(file_path, self)
        self.worker.success.connect(self.on_success)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_success(self, data):
        self.btn_select.setEnabled(True)
        self.panel_result.set_image_bytes(data["result_bytes"])
        self.status_label.setText(f"推理完成，用时 {data['elapsed_ms']} ms")

        summary = data["summary"]
        if summary["total"] == 0:
            self.result_text.setHtml(
                "<p style='font-size:16px;color:#2f6fed'><b>未检测到缺陷或障碍物</b></p>"
                "<p style='color:#888'>可尝试换一张更清晰的路面图片。</p>"
            )
            return

        rows = "".join(
            f"<tr><td style='padding:4px 8px'>{name}</td>"
            f"<td style='padding:4px 8px;text-align:right;color:#2f6fed'><b>{count}</b></td></tr>"
            for name, count in sorted(summary["counts"].items(), key=lambda kv: -kv[1])
        )
        self.result_text.setHtml(
            f"<p style='font-size:16px;margin-bottom:6px'>"
            f"共检出 <b style='color:#2f6fed'>{summary['total']}</b> 个目标</p>"
            f"<table width='100%' cellspacing='0'>{rows}</table>"
        )

    def on_failed(self, msg):
        self.btn_select.setEnabled(True)
        self.status_label.setText("检测失败")
        self.panel_result.image_label.setText("检测失败")
        self.result_text.setHtml(f"<p style='color:#d93025'>{msg}</p>")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RoadDetectWindow()
    win.show()
    sys.exit(app.exec_())
