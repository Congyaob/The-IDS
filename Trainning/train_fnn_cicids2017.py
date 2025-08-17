
import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

data_path = "F:/Project/Dataset/MachineLearningCVE/cicids2017_combined.csv"
scaler_path = "F:/Project/Dataset/MachineLearningCVE/scaler.pkl"
encoder_path = "F:/Project/Dataset/MachineLearningCVE/label_encoder.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU：" if torch.cuda.is_available() else " CPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

df = pd.read_csv(data_path)

df.columns = df.columns.str.strip().str.lower()

print(f"The original data is loaded and the number of samples is：{len(df)}, Number of features：{df.shape[1] - 1}")

if "source_file" in df.columns:
    df = df.drop(columns=["source_file"])

df = df.dropna()

X = df.drop(columns=["label"])
y = df["label"]

X = X.astype("float32")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.long)

from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

class IDSData(Dataset):
    def __init__(self, features, labels):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

train_dataset = IDSData(X_train, y_train)
test_dataset = IDSData(X_test, y_test)

class_sample_count = np.array([np.sum(y_train == t) for t in classes])
weights = 1. / class_sample_count
samples_weights = weights[y_train]
sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

train_loader = DataLoader(train_dataset, batch_size=1024, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

class FNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

model = FNN(input_size=X.shape[1], num_classes=len(classes)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
criterion = nn.CrossEntropyLoss(weight=class_weights)

epochs = 100
loss_history = []
start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "fnn_trained_cicids2017.pth")
print(f"Training completed, time consuming {time.time() - start_time:.1f} 秒")

plt.plot(loss_history)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_curve_cicids2017.png")
print("Saved：loss_curve_cicids2017.png")
