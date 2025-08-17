# The‑IDS: Lightweight Real‑time Intrusion Detection System (FNN-Based)

A lightweight and efficient real‑time intrusion detection system (IDS) built on a feedforward neural network (FNN), capable of fast packet-level inference and deployment in resource-constrained environments (e.g., edge devices via WSL).

##  Repository Structure

```
The‑IDS/
├── DataSet/                     # Raw datasets and preprocessing scripts (demonstrative / external due to privacy)
├── models/                      # Trained model weights
│   ├── advanced_fnn_best_cleaned.pth
│   └── advanced_fnn_final_cleaned.pth
├── results/                     # Evaluation outputs and visualizations
│   ├── loss_curve_advanced_fnn_cleaned.png
│   ├── confusion_matrix_cleaned.png
│   ├── f1_per_class_cleaned.png
│   ├── support_per_class_cleaned.png
│   └── wrong_predictions.csv
├── utils.py                     # Utility functions for feature handling, plotting, etc.
├── train_fnn_cicids2017.py      # Training script using CICIDS2017 dataset
├── train_fnn_v2.py              # Alternative or extended training version
├── evaluate_model.py            # Model evaluation script
├── fnn_scaler.pkl               # Scaler for data normalization
├── fnn_label_encoder.pkl        # Label encoder for mapping class labels
├── advanced_fnn_best_cleaned.pth
├── advanced_fnn_final_cleaned.pth
├── res.py                       # Deployment or inference orchestration script
├── detection_logs.db            # SQLite log file for inference results (generated during runtime)
└── README.md                    # This file
```

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
python res.py
```
Runs the real-time packet capture, feature extraction, FNN inference, and stores predictions into `detection_logs.db`.

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
