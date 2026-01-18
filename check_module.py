import google.generativeai as genai

GOOGLE_API_KEY = "..."
genai.configure(api_key=GOOGLE_API_KEY)

print("Đang hỏi Google danh sách model...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:

    print(f"Lỗi: {e}")
