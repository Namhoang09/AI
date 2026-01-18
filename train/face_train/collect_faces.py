import cv2
import os

name = input("Nhập tên người cần thu ảnh: ")
save_dir = f"dataset/dataset_faces/{name}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])

print("Nhấn Enter để chụp hoặc q để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Enter key
        cv2.imwrite(f"{save_dir}/{count}.jpg", frame)
        print(f"Đã lưu ảnh {count}.jpg")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()