import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Giả lập dữ liệu từ báo cáo của bạn (Hình 3 trong PDF)
# Các nhãn: Nam, Spk26, Spk27, Spk28, Spk29
labels = ["Nam", "Spk26", "Spk27", "Spk28", "Spk29"]

# Ma trận nhầm lẫn giả định (Dựa trên precision/recall trong báo cáo)
# Hàng dọc: Thực tế (True) - Hàng ngang: Dự đoán (Predicted)
cm = np.array([
    [28,  0,  0,  0,  0],  # Nam (1.00)
    [0, 175,  1,  1,  0],  # Spk26 (0.99 - có nhầm lẫn nhỏ)
    [0,  0, 184,  0,  0],  # Spk27 (1.00)
    [0,  2,  0, 232,  0],  # Spk28 (0.99)
    [0,  0,  0,  0, 120],  # Spk29 (1.00)
])

# Chuyển thành DataFrame để vẽ
df_cm = pd.DataFrame(cm, index=labels, columns=labels)

# Vẽ Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')

plt.title('Ma trận nhầm lẫn - Mô hình nhận diện giọng nói (MLP)', fontsize=15)
plt.ylabel('Nhãn thực tế (True Label)', fontsize=12)
plt.xlabel('Nhãn dự đoán (Predicted Label)', fontsize=12)

plt.savefig('./testing/confusion_matrix.png', dpi=300)
plt.show()