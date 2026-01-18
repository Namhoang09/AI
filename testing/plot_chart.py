import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file csv vừa chạy được ở Bước 1
data = pd.read_csv("performance_video.csv")

# Lấy 100 frame đầu tiên hoặc 1 khoảng thời gian ổn định để vẽ cho đẹp
subset = data['FPS'].iloc[10:110]  # Bỏ 10 frame đầu lúc khởi động máy chưa ổn định

plt.figure(figsize=(10, 5))
plt.plot(subset, label='Real-time FPS', color='#007acc', linewidth=2)

# Trang trí biểu đồ
plt.title('Biểu đồ độ ổn định FPS theo thời gian thực', fontsize=14)
plt.xlabel('Thứ tự khung hình (Frame Index)', fontsize=12)
plt.ylabel('Tốc độ khung hình (FPS)', fontsize=12)
plt.axhline(y=data['FPS'].mean(), color='r', linestyle='--', label=f'Trung bình: {data["FPS"].mean():.2f} FPS')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Lưu ảnh để chèn vào báo cáo
plt.savefig('./testing/fps_chart_report.png', dpi=300)
plt.show()