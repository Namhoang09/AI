import torch
import torch.nn as nn
import joblib
import numpy as np
import librosa

# Định nghĩa class MLP (Bắt buộc phải có để load model)
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.model(x)

# Biến toàn cục
voice_model = None
voice_encoder = None
voice_scaler = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_voice_models():
    global voice_model, voice_encoder, voice_scaler
    try:
        voice_encoder = joblib.load("model/voice_model/encoder.pkl")
        voice_scaler = joblib.load("model/voice_model/scaler.pkl")

        input_dim = voice_scaler.mean_.shape[0]
        num_classes = len(voice_encoder.classes_)

        voice_model = MLP(input_dim, num_classes).to(device)
        voice_model.load_state_dict(torch.load("model/voice_model/mlp_model.pth", map_location=device))
        voice_model.eval()
        print("Model giọng nói (MLP) tải xong")
        return True
    except Exception as e:
        print(f"Lỗi tải model giọng nói: {e}")
        return False

def extract_features(audio, sr=16000):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0)
    feat_summary = np.concatenate([np.mean(features, axis=1), np.std(features, axis=1)])
    return feat_summary

def predict_voice(audio, sr=16000):
    if voice_model is None:
        return "Người lạ", 0.0

    audio = librosa.util.normalize(audio)
    features = extract_features(audio, sr)
    features_scaled = voice_scaler.transform([features])
    x_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = voice_model(x_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_index = np.argmax(probs)
    confidence = probs[pred_index]

    if confidence < 0.7:
        name = "Người lạ"
    else:
        name = voice_encoder.inverse_transform([pred_index])[0]
    return name, confidence