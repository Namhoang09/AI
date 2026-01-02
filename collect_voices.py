import sounddevice as sd
from scipy.io.wavfile import write
import os

name = input("Nhập tên người cần ghi âm: ")
save_dir = f"dataset/dataset_voices/{name}"
os.makedirs(save_dir, exist_ok=True)

sample_rate = 16000
duration = 4  # giây

existing_files = [f for f in os.listdir(save_dir) if f.endswith('.wav')]
if existing_files:
    numbers = []
    for f in existing_files:
        base = f.split('.')[0]
        if base.isdigit():
            numbers.append(int(base))
    count = max(numbers) + 1 if numbers else 0
else:
    count = 0

while True:
    cmd = input("Nhấn Enter để ghi âm hoặc q để thoát")
    if cmd.lower() == 'q':
        print("Dừng ghi âm")
        break

    print("Đang ghi âm, hãy nói vào micro")
    audio = sd.rec(duration * sample_rate, samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()

    file_path = f"{save_dir}/{count}.wav"
    write(file_path, sample_rate, audio)
    print(f"Đã ghi xong: {file_path}")
    count += 1
