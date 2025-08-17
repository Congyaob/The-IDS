
import scapy.all as scapy
import numpy as np
import requests
import time
from datetime import datetime
from db_logger import init_db, insert_log

init_db()

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

def extract_features(packet):
    try:
        features = [
            len(packet),                           
            int(packet.time % 1000),              
            int(packet.haslayer("TCP")),           
            int(packet.haslayer("UDP")),          
            int(packet.haslayer("ICMP")),         
            int(packet.haslayer("Raw"))           
        ]
        while len(features) < 78:
            features.append(0)
        return features[:78]
    except Exception as e:
        print(" Feature extraction failed:", e)
        return None

def send_to_model(features):
    try:
        res = requests.post(PREDICT_URL, json={'features': features})
        result = res.json()

        prediction = result.get("prediction", "Unknown")
        label = CLASS_LABELS.get(prediction, prediction)
        confidences = result.get("confidence", [])
        confidence_score = max(confidences) if confidences else 0.0

        print(f"\n Captured Packet at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Prediction: {label} ({prediction})")
        print(f" Confidence: {confidence_score * 100:.2f}%")

        if prediction != "Class0" and confidence_score > CONFIDENCE_THRESHOLD:
            print(" ALERT! Suspicious activity detected!")

        insert_log(
            timestamp=datetime.now().isoformat(),
            features=features,
            prediction=label,
            confidence=round(confidence_score, 4)
        )

        return result
    except Exception as e:
        print(" Predict request failed:", e)
        return None

def process_packet(packet):
    features = extract_features(packet)
    if features:
        result = send_to_model(features)
        if result and result['prediction'] != "Class0" and max(result['confidence']) > CONFIDENCE_THRESHOLD:
            print(f" ALERT! Detected: {result['prediction']}")

if __name__ == '__main__':
    print(" Real-time packet inspection started...")
    scapy.sniff(prn=process_packet, store=False)
