import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import json
import os

model_path = r"F:\Project\AI\advanced_fnn_final_cleaned.pth"
scaler_path = r"F:\Project\AI\fnn_scaler.pkl"
encoder_path = r"F:\Project\AI\fnn_label_encoder.pkl"
test_csv_path = r"F:\Project\Dataset\MachineLearningCVE\test_cleaned_scaled.csv"

class AdvancedFNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

encoder = joblib.load(encoder_path)
scaler = joblib.load(scaler_path)

df = pd.read_csv(test_csv_path)
X = df.drop(columns=["Label"]).values
y_true = df["Label"].astype(str).values

known_labels = set(encoder.classes_)
mask = np.array([label in known_labels for label in y_true])
X = X[mask]
y_true = y_true[mask]

X_scaled = scaler.transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_encoded = encoder.transform(y_true)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

input_size = X.shape[1]
num_classes = len(encoder.classes_)
model = AdvancedFNN(input_size, num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

with torch.no_grad():
    outputs = model(X_tensor.to(device))
    y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

report = classification_report(
    y_encoded,
    y_pred,
    target_names=encoder.classes_,
    zero_division=0,
    output_dict=True
)

with open("classification_report_cleaned.json", "w") as f:
    json.dump(report, f, indent=4)
print(" Classification report saved as classification_report_cleaned.json")

macro_f1 = report["macro avg"]["f1-score"]
weighted_f1 = report["weighted avg"]["f1-score"]
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")

decoded_preds = encoder.inverse_transform(y_pred)
df_wrong = pd.DataFrame({
    "True Label": y_true,
    "Predicted Label": decoded_preds
})
df_wrong["Match"] = df_wrong["True Label"] == df_wrong["Predicted Label"]
df_wrong[~df_wrong["Match"]].to_csv("wrong_predictions.csv", index=False)
print(" Error predictions are exported as wrong_predictions.csv")

# ========== 可视化 ==========
cm = confusion_matrix(y_encoded, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap="Blues", fmt="d", square=True, cbar=False)
plt.title("Confusion Matrix (Cleaned FNN)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_cleaned.png")
plt.close()

f1_scores = [report[label]["f1-score"] for label in encoder.classes_]
plt.figure(figsize=(12, 5))
plt.bar(encoder.classes_, f1_scores, color="skyblue")
plt.xticks(rotation=90)
plt.ylabel("F1 Score")
plt.title("Per-Class F1 Score (Cleaned FNN)")
plt.tight_layout()
plt.savefig("f1_per_class_cleaned.png")
plt.close()

support = [report[label]["support"] for label in encoder.classes_]
plt.figure(figsize=(12, 5))
plt.bar(encoder.classes_, support, color="orange")
plt.xticks(rotation=90)
plt.ylabel("Support")
plt.title("Per-Class Sample Support (Cleaned FNN)")
plt.tight_layout()
plt.savefig("support_per_class_cleaned.png")
plt.close()

print("Chart saved：confusion_matrix_cleaned.png, f1_per_class_cleaned.png, support_per_class_cleaned.png")
