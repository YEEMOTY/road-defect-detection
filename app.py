import os
import time
import uuid
import threading

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, jsonify, render_template, request, url_for
from ultralytics import YOLO

# ==================== 配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "best.pt"))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "upload")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "output")
# 默认置信度阈值，可通过请求参数 conf 覆盖
DEFAULT_CONF = float(os.environ.get("CONF_THRESHOLD", "0.25"))
# 允许的图片扩展名
ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_FILE_MB = 16
# 公网地址前缀（cpolar 穿透时在环境变量里配置，例如 https://xxx.cpolar.io）
PUBLIC_PREFIX = os.environ.get("PUBLIC_PREFIX", "").rstrip("/")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024
app.json.ensure_ascii = False  # 返回中文而不是 \uXXXX

# ==================== 模型 ====================
# 推理服务同时只服务一个请求，加锁避免多线程共享模型出错
model = YOLO(MODEL_PATH)
model_lock = threading.Lock()

# 中文标签需要 TrueType 字体，OpenCV 自带字体无法显示中文
FONT_PATH = os.environ.get("LABEL_FONT", r"C:\Windows\Fonts\msyh.ttc")
if not os.path.exists(FONT_PATH):
    for fallback in (r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if os.path.exists(fallback):
            FONT_PATH = fallback
            break

# 每个类别一个固定颜色（BGR 顺序的 RGB 值）
PALETTE = [
    (255, 86, 86), (255, 156, 66), (250, 214, 66), (140, 216, 90), (66, 200, 178),
    (72, 160, 255), (126, 112, 255), (214, 96, 214), (255, 118, 176), (110, 190, 255),
    (176, 216, 66), (66, 138, 120), (255, 196, 140), (150, 150, 150), (255, 92, 92),
    (96, 200, 96), (88, 190, 255), (200, 170, 90), (170, 120, 255),
]


def get_font(size):
    """按图片尺寸自适应字号"""
    size = max(14, int(size))
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def draw_detections(bgr_img, boxes, names, min_font=14):
    """用 PIL 在图上绘制检测框和中文类别标签（避免 OpenCV 中文乱码）"""
    img = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    h, w = bgr_img.shape[:2]
    font = get_font(max(min_font, h // 40))
    line_width = max(2, h // 320)

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cls_id = int(box.cls[0])
        color = PALETTE[cls_id % len(PALETTE)]
        label = str(names[cls_id])

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # 标签默认放在框上方，空间不足时放到框内
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw, th = right - left, bottom - top
        pad = line_width + 2
        lx, ly = x1, y1 - th - 2 * pad
        if ly < 0:
            lx, ly = x1, y1
        lx = min(lx, w - tw - 2 * pad)
        lx = max(lx, 0)

        draw.rectangle([lx, ly, lx + tw + 2 * pad, ly + th + 2 * pad], fill=color)
        draw.text((lx + pad, ly + pad - top), label, font=font, fill=(255, 255, 255))

    return img


def image_url(relative_path):
    """生成图片对外访问地址：优先用 PUBLIC_PREFIX，否则用当前请求的域名"""
    if PUBLIC_PREFIX:
        return f"{PUBLIC_PREFIX}/{relative_path}"
    return request.host_url.rstrip("/") + url_for("static", filename=relative_path)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"code": 413, "msg": f"图片超过 {MAX_FILE_MB}MB 限制"}), 413


@app.errorhandler(500)
def server_error(_):
    return jsonify({"code": 500, "msg": "服务器内部错误"}), 500


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "code": 200,
        "msg": "服务正常",
        "model": os.path.basename(MODEL_PATH),
        "classes": list(model.names.values()),
    })


@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    file = request.files.get("image")
    if file is None or not file.filename:
        return jsonify({"code": 400, "msg": "未上传图片"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXT:
        return jsonify({"code": 400, "msg": f"不支持的图片格式：{ext or '未知'}"}), 400

    try:
        conf = float(request.form.get("conf", DEFAULT_CONF))
    except ValueError:
        conf = DEFAULT_CONF
    conf = min(max(conf, 0.01), 0.99)

    # 用 uuid 命名，原件与结果图分开保存
    name = uuid.uuid4().hex
    orig_name = f"{name}.{ext}"
    orig_path = os.path.join(UPLOAD_DIR, orig_name)

    try:
        file.save(orig_path)
        # 校验是不是真正的图片文件
        img = cv2.imdecode(np.fromfile(orig_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            os.remove(orig_path)
            return jsonify({"code": 400, "msg": "文件不是有效的图片"}), 400

        with model_lock:
            res = model.predict(img, conf=conf, verbose=False)[0]

        plotted = draw_detections(img, res.boxes, res.names)
        result_name = f"{name}_result.jpg"
        plotted.save(os.path.join(OUTPUT_DIR, result_name), quality=92)

        detections = []
        counts = {}
        for box in res.boxes:
            cls_name = str(res.names[int(box.cls[0])])
            detections.append({
                "class": cls_name,
                "conf": round(float(box.conf[0]), 3),
                "bbox": [int(v) for v in box.xyxy[0].tolist()],
            })
            counts[cls_name] = counts.get(cls_name, 0) + 1

        return jsonify({
            "code": 200,
            "msg": "推理成功",
            "data": {
                "original_url": image_url(f"upload/{orig_name}"),
                "image_url": image_url(f"output/{result_name}"),
                "detections": detections,
                "summary": {"total": len(detections), "counts": counts},
                "elapsed_ms": int((time.time() - start) * 1000),
            },
        })
    except Exception as err:
        app.logger.exception("推理失败")
        return jsonify({"code": 500, "msg": f"推理失败：{err}"}), 500


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
        threaded=True,
    )
