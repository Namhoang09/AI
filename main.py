import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

import cv2
import sounddevice as sd
import numpy as np
import time
import signal 

import server_app
from model.gemini_bot import ask_gemini
from model.system_utils import handle_shutdown, handle_stop_server_event

from train.face_train.face_inference import load_face_models, recognize_face
from train.voice_train.voice_inference import load_voice_models, predict_voice
from train.voice_train.stt_inference import load_whisper_model, get_text
from deepface import DeepFace 
from flask import render_template, Response

# KHỞI ĐỘNG HỆ THỐNG
load_face_models()
load_voice_models()
load_whisper_model()

# LOGIC LUỒNG 1: CAMERA
def generate_frames():
    cap = cv2.VideoCapture(0)
    last_check_time = 0
    current_names = []
    
    while not server_app.stop_event.is_set():
        ret, frame = cap.read()
        if not ret: break
            
        try:
            faces = DeepFace.extract_faces(frame, enforce_detection=False, detector_backend='opencv')
        except Exception:
            faces = []

        if time.time() - last_check_time >= 1 and len(faces) > 0:
            new_names = []
            try:
                for face in faces: 
                    fa = face["facial_area"]
                    x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
                    pad = 10 
                    cropped_face = frame[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
                    if cropped_face.size == 0: 
                        continue

                    results = DeepFace.represent(
                        img_path=cropped_face, 
                        model_name="ArcFace", 
                        enforce_detection=False
                    )
                    
                    embedding = results[0]["embedding"]
                    name = recognize_face(embedding) 
                    new_names.append(name)
                current_names = new_names
            except Exception: pass
            last_check_time = time.time()

        # Vẽ khung
        for i, face in enumerate(faces):
            fa = face["facial_area"]
            x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
            name = current_names[i] if i < len(current_names) else "..."
            color = (0, 255, 0) if name != "Unknown" and name != "..." else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

# LOGIC LUỒNG 2: GIỌNG NÓI & TRỢ LÝ ẢO
voice_buffer = []
sr = 16000
chunk_size = int(sr * 1)
window_size = sr * 4
SILENCE_THRESHOLD = 0.01

def audio_callback(indata, frames, time, status):
    global voice_buffer
    if status: print(status)
    voice_buffer.extend(indata[:, 0])

def run_voice_recognition():
    global voice_buffer
    print("Đang bật microphone...")
    last_response_time = 0
    
    try:
        # Gán vào biến toàn cục trong server_app
        server_app.voice_stream = sd.InputStream(callback=audio_callback, channels=1, blocksize=chunk_size, samplerate=sr)
        
        with server_app.voice_stream:
            while not server_app.stop_event.is_set():
                if len(voice_buffer) < window_size:
                    server_app.socketio.sleep(0.01)
                    continue
                
                audio_chunk = np.array(voice_buffer[:window_size])
                voice_buffer = voice_buffer[window_size:]
                
                if np.sqrt(np.mean(audio_chunk**2)) < SILENCE_THRESHOLD: continue

                try:
                    name, confidence = predict_voice(audio_chunk, sr)
                    text = get_text(audio_chunk, sr)

                    server_app.socketio.emit('new_transcript', {'name': name, 'text': text, 'confidence': float(confidence)})

                    # --- LOGIC TRẢ LỜI ---
                    if text and len(text) > 2 and (time.time() - last_response_time > 3):
                        text_lower = text.lower()
                        
                        # Xử lý lệnh tắt (Offline)
                        if "tắt hệ thống" in text_lower:
                            server_app.socketio.emit('ai_response', {'text': "Tạm biệt! Đang tắt hệ thống"})
                            time.sleep(1)
                            handle_stop_server_event() # Gọi hàm tắt từ module
                            return
                        
                        # Xử lý hỏi đáp (Gemini)
                        else:
                            server_app.socketio.emit('update_status', {'state': 'thinking'})
                            response = ask_gemini(text)
                            server_app.socketio.emit('ai_response', {'text': response})
                            server_app.socketio.emit('update_status', {'state': 'listening'})
                            last_response_time = time.time()

                except Exception as e:
                    print(f"Lỗi xử lý mic: {e}")
    except Exception: pass

# ROUTES & EVENTS
@server_app.app.route('/')
def index(): return render_template('index.html')

@server_app.app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@server_app.socketio.on('connect')
def handle_connect():
    if not hasattr(server_app.app, 'voice_thread_started'):
        server_app.app.voice_thread_started = True
        server_app.socketio.start_background_task(target=run_voice_recognition)

# Đăng ký sự kiện tắt từ module system_utils
@server_app.socketio.on('stop_server')
def on_stop_server():
    handle_stop_server_event()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handle_shutdown)
    
    print("Server: http://127.0.0.1:5000")
    # Chạy server từ module server_app
    server_app.socketio.run(server_app.app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)