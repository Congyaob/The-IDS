# The‑IDS: Lightweight Real‑time Intrusion Detection System (FNN-Based)

A lightweight and efficient real‑time intrusion detection system (IDS) built on a feedforward neural network (FNN), capable of fast packet-level inference and deployment in resource-constrained environments (e.g., edge devices via WSL).

##  Repository Structure

# Lightweight Feedforward IDS (The-IDS)

This repository contains a lightweight real-time Intrusion Detection System (IDS) based on a Feedforward Neural Network (FNN). The project includes model training scripts, real-time feature extraction using Scapy, a Flask-based backend for model inference and alert logging, and a browser-based frontend for visualization.

## 📁 Project Structure

```bash
├── Trainning/                     # Model training and evaluation scripts
│   ├── train_fnn_cicids2017.py    # Main training script using CICIDS2017
│   ├── evaluate_model.py          # Evaluation script for trained model
│   └── merge.py                   # Dataset merging utility (if needed)

├── ids/                           # Core backend codebase
│   ├── app.py                     # Main Flask application
│   ├── app_diag1.py               # Diagnostic endpoints (optional)
│   ├── app_probe.py               # Probe test module
│   ├── scaler.pkl                 # Trained scaler for preprocessing
│   ├── backend/
│   │   ├── model/
│   │   │   └── advanced_fnn_best_cleaned.pth  # Trained PyTorch model
│   │   └── templates/
│   │       └── index.html         # HTML template for backend
│   └── __pycache__/               # Python bytecode cache

├── frontend/                      # Web-based frontend for real-time display
│   ├── index.html                 # Main UI
│   ├── script.js                  # JavaScript logic
│   ├── style.css                  # CSS styling
│   ├── db_logger.py               # Logs detection to database
│   ├── detections.db              # SQLite database for storing alerts
│   ├── detection_logs.db          # Alternate database
│   └── feature_extractor.py      # Scapy-based feature extraction

└── README.md                      # Project documentation


##  Prerequisites

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Recommended environments: WSL2, Linux, or macOS with Scapy support.

##  Usage Overview

### 1. Train the Model
```bash
python train_fnn_cicids2017.py
```
This script handles dataset preprocessing, model training, and saving of the best-performing model weights (`.pth`), scaler, and label encoder.

### 2. Evaluate Performance
```bash
python evaluate_model.py
```
Generates evaluation metrics and visualizations (e.g., confusion matrix, per-class F1 score).

### 3. Real‑Time Deployment (Inference)
```bash
cd ids/
python app.py
```
Then visit: http://127.0.0.1:5000 to view detection results.
Feature extraction runs automatically using feature_extractor.py to capture live traffic and generate model input.

##  Highlights

- **Fast inference:** Deploys FNN model with ~50 ms latency under typical home network traffic.
- **Lightweight design:** No recurrent layers; efficient for edge deployment.
- **Real-time alerts:** Detects and logs anomalies with timestamp, source/destination IP, predicted class, and confidence.

##  Future Enhancements

- Support for feature-level data fusion from multiple datasets.
- Ensemble models or temporal smoothing to improve detection of low-frequency attacks.
- Web-based dashboard for live visualization and alert workflows.

##  License

Licensed under [MIT](LICENSE) – feel free to reuse and build upon this work.

##  Contact

Repository: [https://github.com/Congyaob/The-IDS](https://github.com/Congyaob/The-IDS)  
Questions? Reach out at: `a1880824@adelaide.edu.au`
