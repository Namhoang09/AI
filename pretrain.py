import librosa
import soundfile as sf
import os
import numpy as np
import re

source = "dataset/dataset_voices/Nam"
target = source

os.makedirs(target, exist_ok=True)

for file in os.listdir(source):
    if not file.endswith(".wav"):
        continue

    # tránh augment những file đã là bản augment (tránh vòng lặp vô tận)
    if re.search(r'(_p1|_slow|_n)\.wav$', file):
        continue

    path = os.path.join(source, file)
    try:
        y, sr = librosa.load(path, sr=16000)
        # chuẩn hoá biên độ trước khi augment
        y = librosa.util.normalize(y)

        # Augment 1: tăng cao độ nhẹ
        y_shifted = librosa.effects.pitch_shift(y, n_steps=1.0, sr=sr)
        sf.write(os.path.join(target, file.replace(".wav", "_p1.wav")), y_shifted, sr)

        # Augment 2: giảm tốc độ (time stretch)
        y_stretch = librosa.effects.time_stretch(y, rate=0.9)
        sf.write(os.path.join(target, file.replace(".wav", "_slow.wav")), y_stretch, sr)

        # Augment 3: thêm noise nhẹ
        noise = np.random.normal(0, 0.005, y.shape)
        sf.write(os.path.join(target, file.replace(".wav", "_n.wav")), (y + noise), sr)

        print(f"Đã tạo augment cho: {file}")

    except Exception as e:
        print(f"Lỗi khi xử lý {path}: {e}")
