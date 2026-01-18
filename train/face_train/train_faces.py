import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import joblib
from deepface import DeepFace

dataset_dir = "dataset/dataset_faces"
all_embeddings_list = []
all_labels_list = []

print("Đang tạo embedding cho database vui lòng chờ")
for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_dir):
        continue

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        try:
            result = DeepFace.represent(
                img_path=img_path,
                model_name="ArcFace",
                enforce_detection=False
            )
            embedding = result[0]["embedding"]
            all_embeddings_list.append(embedding)
            all_labels_list.append(person)

        except Exception as e:
            print("Bỏ qua ảnh:", img_path, " Lỗi:", e)

    if len(all_embeddings_list) > len(all_labels_list) - len(os.listdir(person_dir)):
        print(f"{person}: có {len(os.listdir(person_dir))} ảnh, đã xử lý")

all_embeddings_matrix = np.array(all_embeddings_list)
os.makedirs("model/face_model", exist_ok=True)

np.save("model/face_model/all_embeddings.npy", all_embeddings_matrix)
joblib.dump(all_labels_list, "model/face_model/all_labels.pkl")

print("\nĐã tạo và lưu database khuôn mặt thành công")