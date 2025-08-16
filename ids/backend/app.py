# backend/app.py
from __future__ import annotations
from pathlib import Path
import os
import json
import sqlite3

from flask import Flask, request, jsonify, send_file, send_from_directory
import numpy as np
import torch
import torch.nn as nn
import joblib

# ------------------------------
# 路径与 Flask 实例
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent                  # .../ids/backend
PROJECT_ROOT = BASE_DIR.parent                              # .../ids
FRONTEND_DIR = (PROJECT_ROOT / "frontend").resolve()        # .../ids/frontend
INDEX_HTML = FRONTEND_DIR / "index.html"

MODEL_PATH = BASE_DIR / "model" / "advanced_fnn_best_cleaned.pth"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# 数据库路径：优先环境变量 IDS_DB_PATH；否则尝试项目根目录 detections.db，再回退到 backend/detections.db
_env_db = os.environ.get("IDS_DB_PATH", "").strip()
if _env_db:
    DB_PATH = Path(_env_db).resolve()
else:
    cand1 = PROJECT_ROOT / "detections.db"
    cand2 = BASE_DIR / "detections.db"
    DB_PATH = cand1 if cand1.exists() else cand2
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

print("🛠 FRONTEND_DIR:", FRONTEND_DIR)
print("🛠 INDEX_HTML  :", INDEX_HTML)
print("🗄  DB_PATH     :", DB_PATH)

# 一定先创建 app，再写 @app.route
app = Flask(__name__, static_folder=None)


# ------------------------------
# 前端页面与静态资源
# ------------------------------
@app.route("/")
def index():
    if not INDEX_HTML.exists():
        return f"index.html not found at: {INDEX_HTML}", 404
    return send_file(str(INDEX_HTML))

@app.route("/static/<path:path>")
def static_files(path: str):
    return send_from_directory(str(FRONTEND_DIR), path)

@app.route("/favicon.ico")
def favicon():
    # 可自行放一个 favicon 到 frontend 里；这里先返回 204
    return ("", 204)


# ------------------------------
# 模型与推理
# ------------------------------
INPUT_SIZE = 78
NUM_CLASSES = 15
LABELS = [
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "Bot",
    "Infiltration",
    "Web Attack – Brute Force",
    "Web Attack – XSS",
    "Web Attack – SQL Injection",
    "DDoS",
    "FTP-Patator",
    "SSH-Patator",
    "DoS GoldenEye",
    "DoS Slowloris",
    "DoS Slowhttptest",
    "Heartbleed",
]

model = nn.Sequential(
    nn.Linear(INPUT_SIZE, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(), nn.Dropout(0.3),
    nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.LeakyReLU(), nn.Dropout(0.3),
    nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES),
)

def _load_weights_safely(path: Path):
    if not path.exists():
        print("❌ Model not found:", path)
        return
    try:
        raw = torch.load(str(path), map_location="cpu", weights_only=True)  # 新版推荐
    except TypeError:
        raw = torch.load(str(path), map_location="cpu")  # 兼容旧版
    state = raw
    if isinstance(raw, dict) and isinstance(raw.get("state_dict"), dict):
        state = raw["state_dict"]
    if isinstance(raw, dict) and isinstance(raw.get("model"), dict):
        state = raw["model"]
    cleaned = {
        (k.replace("model.", "", 1) if isinstance(k, str) and k.startswith("model.") else k): v
        for k, v in state.items()
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print("⚠️ Missing keys:", missing)
    if unexpected:
        print("⚠️ Unexpected keys:", unexpected)
    model.eval()
    print("✅ Model loaded:", path.name)

_load_weights_safely(MODEL_PATH)

scaler = None
if SCALER_PATH.exists():
    try:
        scaler = joblib.load(str(SCALER_PATH))
        print("✅ Scaler loaded")
    except Exception as e:
        print("⚠️ Scaler load failed:", e)
else:
    print("ℹ️ No scaler file, continue without scaler")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        feats = payload.get("features")
        if not isinstance(feats, (list, tuple)) or len(feats) != INPUT_SIZE:
            return jsonify({"error": f"expected {INPUT_SIZE} features"}), 400

        X = np.array(feats, dtype=np.float32).reshape(1, -1)
        if scaler is not None:
            X = scaler.transform(X)

        with torch.no_grad():
            logits = model(torch.from_numpy(X))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        return jsonify({
            "prediction": f"Class{idx}",
            "prediction_label": LABELS[idx],
            "confidence": probs.tolist(),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------
# 日志（实时流接口供前端轮询）
# 需要表结构类似：
#   CREATE TABLE logs (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     ts TEXT,
#     features TEXT,
#     prediction TEXT,
#     confidence TEXT
#   );
# ------------------------------
def _dict_row(cursor, row):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

def _fetch_logs(since_id: int = 0, limit: int = 100):
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = _dict_row
    cur = conn.cursor()
    cur.execute("""
        SELECT id, ts, prediction, confidence
        FROM logs
        WHERE id > ?
        ORDER BY id ASC
        LIMIT ?
    """, (since_id, limit))
    rows = cur.fetchall()
    conn.close()
    return rows

@app.route("/logs")
def logs():
    """
    轮询：
      GET /logs?since=<last_id>&limit=100
    返回：
      { "items":[{id, ts, prediction, prediction_label, max_conf}], "last_id": <id> }
    """
    try:
        since = int(request.args.get("since", "0"))
        limit = int(request.args.get("limit", "100"))
    except Exception:
        since, limit = 0, 100

    try:
        items = _fetch_logs(since_id=since, limit=limit)
    except Exception as e:
        return jsonify({"items": [], "last_id": since, "error": str(e)}), 200

    out = []
    for r in items:
        pred = r.get("prediction", "Class0")
        # 解析 confidence：可能是 JSON 字符串
        conf_raw = r.get("confidence", [])
        if isinstance(conf_raw, str):
            try:
                conf_list = json.loads(conf_raw)
            except Exception:
                conf_list = []
        elif isinstance(conf_raw, (list, tuple)):
            conf_list = conf_raw
        else:
            conf_list = []

        if isinstance(pred, str) and pred.startswith("Class"):
            try:
                idx = int(pred.replace("Class", "") or 0)
            except Exception:
                idx = 0
        else:
            idx = 0

        label = LABELS[idx] if 0 <= idx < len(LABELS) else pred
        max_conf = float(max(conf_list)) if conf_list else 0.0
        out.append({
            "id": int(r["id"]),
            "ts": r["ts"],
            "prediction": pred,
            "prediction_label": label,
            "max_conf": max_conf,
        })

    last_id = out[-1]["id"] if out else since
    return jsonify({"items": out, "last_id": last_id})

@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200


# ------------------------------
# 启动
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
