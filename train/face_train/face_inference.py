import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Biến toàn cục lưu model
all_embeddings = None
all_labels = None

def load_face_models():
    global all_embeddings, all_labels
    try:
        # Đường dẫn tính từ thư mục gốc (nơi chạy main_web.py)
        all_embeddings = np.load("model/face_model/all_embeddings.npy")
        all_labels = joblib.load("model/face_model/all_labels.pkl")
        print("Model khuôn mặt tải xong")
        return True
    except Exception as e:
        print(f"Lỗi tải model khuôn mặt: {e}")
        return False

def recognize_face(face_embedding, threshold=0.7):
    if all_embeddings is None:
        return "Unknown"

    sims = cosine_similarity([face_embedding], all_embeddings)[0]
    best_index = np.argmax(sims)
    best_score = sims[best_index]
    
    if best_score < threshold:
        return "Unknown"
    else:
        return all_labels[best_index]