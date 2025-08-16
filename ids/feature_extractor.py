# feature_extractor.py

import scapy.all as scapy
import numpy as np
import requests
import time
from datetime import datetime
from db_logger import init_db, insert_log

# 初始化数据库
init_db()

# 类别编号 → 可读标签映射
CLASS_LABELS = {
    "Class0": "BENIGN",
    "Class1": "DoS Hulk",
    "Class2": "PortScan",
    "Class3": "Bot",
    "Class4": "Infiltration",
    "Class5": "Web Attack – Brute Force",
    "Class6": "Web Attack – XSS",
    "Class7": "Web Attack – SQL Injection",
    "Class8": "DDoS",
    "Class9": "FTP-Patator",
    "Class10": "SSH-Patator",
    "Class11": "DoS GoldenEye",
    "Class12": "DoS Slowloris",
    "Class13": "DoS Slowhttptest",
    "Class14": "Heartbleed"
}

PREDICT_URL = 'http://127.0.0.1:5000/predict'
CONFIDENCE_THRESHOLD = 0.95

# 特征提取函数（需根据训练特征结构补充完善）
def extract_features(packet):
    try:
        features = [
            len(packet),                           # 特征1: 包长度
            int(packet.time % 1000),               # 特征2: 时间戳模1000
            int(packet.haslayer("TCP")),           # 特征3: TCP
            int(packet.haslayer("UDP")),           # 特征4: UDP
            int(packet.haslayer("ICMP")),          # 特征5: ICMP
            int(packet.haslayer("Raw"))            # 特征6: 原始数据
        ]
        while len(features) < 78:
            features.append(0)
        return features[:78]
    except Exception as e:
        print("❌ Feature extraction failed:", e)
        return None

# 模型预测
def send_to_model(features):
    try:
        res = requests.post(PREDICT_URL, json={'features': features})
        result = res.json()

        prediction = result.get("prediction", "Unknown")
        label = CLASS_LABELS.get(prediction, prediction)
        confidences = result.get("confidence", [])
        confidence_score = max(confidences) if confidences else 0.0

        # 打印可读输出
        print(f"\n📦 Captured Packet at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🧠 Prediction: {label} ({prediction})")
        print(f"📈 Confidence: {confidence_score * 100:.2f}%")

        if prediction != "Class0" and confidence_score > CONFIDENCE_THRESHOLD:
            print("🚨 ALERT! Suspicious activity detected!")

        # 写入数据库
        insert_log(
            timestamp=datetime.now().isoformat(),
            features=features,
            prediction=label,
            confidence=round(confidence_score, 4)
        )

        return result
    except Exception as e:
        print("❌ Predict request failed:", e)
        return None

# 抓包主函数
def process_packet(packet):
    features = extract_features(packet)
    if features:
        result = send_to_model(features)
        if result and result['prediction'] != "Class0" and max(result['confidence']) > CONFIDENCE_THRESHOLD:
            print(f"🚨 ALERT! Detected: {result['prediction']}")

if __name__ == '__main__':
    print("🕵️‍♂️ Real-time packet inspection started...")
    scapy.sniff(prn=process_packet, store=False)
