from PIL import Image, ImageFile
import os

# Cho phép load ảnh lỗi bị thiếu dữ liệu
ImageFile.LOAD_TRUNCATED_IMAGES = True

dataset_dir = "dataset/dataset_faces"

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(root, file)
            try:
                with Image.open(path) as img: 
                    img = img.convert("RGB")
                    img.save(path, "JPEG", quality=95, optimize=True)

            except Exception as e:
                print(f"Lỗi khi xử lý {path}: {e}")
                try:
                    os.remove(path)   # xóa file lỗi
                    print(f"Đã xóa {path}")
                except Exception as e2:
                    print(f"Không xóa được {path}: {e2}")

print("Đã chuẩn hóa ảnh trong dataset_faces")
