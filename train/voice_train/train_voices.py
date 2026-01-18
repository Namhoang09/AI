import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import sys
import time

X, y = [], []
dataset_dir = "dataset/dataset_voices"

print("Đang trích xuất đặc trưng MFCC từ dữ liệu")
total_files = sum(len(files) for _, _, files in os.walk(dataset_dir)) 
processed = 0 
start_time = time.time()

for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    for file in os.listdir(person_dir):
        if not file.endswith(".wav"):
            continue
        path = os.path.join(person_dir, file)
        try:
            audio, sr = librosa.load(path, sr=16000)
            audio = librosa.util.normalize(audio)

            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)

            features = np.concatenate([mfcc, delta, delta2], axis=0)
            feat_summary = np.concatenate([np.mean(features, axis=1), np.std(features, axis=1)])
            
            X.append(feat_summary)
            y.append(person)

        except Exception as e:
            print(f"Lỗi khi xử lý {path}: {e}")

        processed += 1 
        percent = processed / total_files * 100 
        sys.stdout.write(f"\rTiến độ: {processed}/{total_files} file ({percent:.1f}%)") 
        sys.stdout.flush()

X = np.array(X)

# ====== Encode labels ======
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ====== Chuẩn hóa ======
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ====== Train/Test split ======
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

input_dim = X_train.shape[1]
num_classes = len(np.unique(y_encoded))

# ====== MLP Model ======
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nĐang huấn luyện MLP")
epochs = 50
batch_size = 32

for epoch in range(epochs):
    model.train()
    perm = torch.randperm(X_train.size(0))
    total_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        idx = perm[i:i+batch_size]
        batch_x, batch_y = X_train[idx].to(device), y_train[idx].to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# ====== Đánh giá ======
model.eval()
with torch.no_grad():
    outputs = model(X_test.to(device))
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

print("\nBáo cáo đánh giá:")
print(classification_report(y_test, preds, target_names=encoder.classes_))

os.makedirs("model/voice_model", exist_ok=True)
torch.save(model.state_dict(), "model/voice_model/mlp_model.pth")
joblib.dump(encoder, "model/voice_model/encoder.pkl")
joblib.dump(scaler, "model/voice_model/scaler.pkl")

elapsed = time.time() - start_time
print(f"\nĐã train xong MLP trong {elapsed:.2f} giây!")