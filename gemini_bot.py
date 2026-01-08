import google.generativeai as genai
from datetime import datetime

# CẤU HÌNH GEMINI
GOOGLE_API_KEY = "..."
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-flash-latest')

def ask_gemini(text):
    try:
        now = datetime.now()
        current_time_str = now.strftime("%H:%M ngày %d/%m/%Y")
        
        prompt = (
            f"Hiện tại là: {current_time_str}. "
            f"Bạn là trợ lý ảo AI thông minh, hữu ích. "
            f"Hãy trả lời ngắn gọn, súc tích (dưới 50 từ) câu hỏi sau bằng tiếng Việt: {text}"
        )
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:

        return f"Xin lỗi, não bộ đang mất kết nối. Lỗi: {e}"
