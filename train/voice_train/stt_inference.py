from faster_whisper import WhisperModel

whisper_model = None

def load_whisper_model():
    global whisper_model
    try:
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("Model Whisper tải xong")
        return True
    except Exception as e:
        print(f"Lỗi tải model Whisper: {e}")
        return False

def get_text(audio_chunk, sr=16000):
    if whisper_model is None:
        return ""
        
    prompt = "Nam, xin chào, mấy giờ, tắt hệ thống."

    segments, info = whisper_model.transcribe(
        audio_chunk,
        beam_size=5,
        language="vi",       
        initial_prompt=prompt
    )

    text = "".join(segment.text for segment in segments)
    return text.strip()