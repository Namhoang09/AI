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
import pandas as pd

import server_app
from model.gemini_bot import ask_gemini
from model.system_utils import handle_shutdown, handle_stop_server_event

from train.face_train.face_inference import load_face_models, recognize_face
from train.voice_train.voice_inference import load_voice_models, predict_voice
from train.voice_train.stt_inference import load_whisper_model, get_text
from deepface import DeepFace 
from flask import render_template, Response

video_logs = []
audio_logs = []

# KHỞI ĐỘNG HỆ THỐNG
load_face_models()
load_voice_models()
load_whisper_model()

def save_perf_logs():
    print("\nĐang lưu dữ liệu hiệu năng ra file CSV...")
    if video_logs:
        df_video = pd.DataFrame(video_logs)
        df_video.to_csv("./testing/performance_video.csv", index=False)
        print(" -> Đã lưu performance_video.csv")
    
    if audio_logs:
        df_audio = pd.DataFrame(audio_logs)
        df_audio.to_csv("./testing/performance_audio.csv", index=False)
        print(" -> Đã lưu performance_audio.csv")

# LOGIC LUỒNG 1: CAMERA
def generate_frames():
    cap = cv2.VideoCapture(0)
    last_check_time = 0
    current_names = []

    frame_count = 0
    start_time = time.time()
    
    while not server_app.stop_event.is_set():
        t_start = time.time()

        ret, frame = cap.read()
        if not ret: break

        t_capture = time.time()
        ai_run = False
        t_preprocess = t_capture
        t_ai = t_capture

        if time.time() - last_check_time >= 1:
            ai_run = True
            try:
                faces = DeepFace.extract_faces(frame, enforce_detection=False, detector_backend='opencv')
                t_preprocess = time.time()

                if len(faces) > 0:
                    new_names = []
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
                else:
                    current_names = []

                t_ai = time.time()
            except Exception:
                faces = []
                t_preprocess = time.time()
                t_ai = time.time()
            
            last_check_time = time.time()
        
        else:
            faces = [] 
            t_preprocess = time.time()
            t_ai = time.time()

        # Vẽ khung
        for i, face in enumerate(faces):
            fa = face["facial_area"]
            x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
            name = current_names[i] if i < len(current_names) else "..."
            color = (0, 255, 0) if name != "Unknown" and name != "..." else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
        ret, buffer = cv2.imencode('.jpg', frame)
        t_render = time.time()

        latency_capture = (t_capture - t_start) * 1000
        latency_pre = (t_preprocess - t_capture) * 1000 if ai_run else 0
        latency_ai = (t_ai - t_preprocess) * 1000 if ai_run else 0
        latency_render = (t_render - t_ai) * 1000
        total_latency = (t_render - t_start) * 1000
        current_fps = 1.0 / (time.time() - t_start) if (time.time() - t_start) > 0 else 0

        video_logs.append({
            "Capture_ms": latency_capture,
            "Preprocess_ms": latency_pre,
            "AI_Inference_ms": latency_ai,
            "Render_ms": latency_render,
            "Total_ms": total_latency,
            "FPS": current_fps,
            "AI_Active": 1 if ai_run else 0
        })

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    save_perf_logs()

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
                
                t_start_audio = time.time()

                audio_chunk = np.array(voice_buffer[:window_size])
                voice_buffer = voice_buffer[window_size:]
                
                if np.sqrt(np.mean(audio_chunk**2)) < SILENCE_THRESHOLD: continue

                t_vad = time.time()

                try:
                    name, confidence = predict_voice(audio_chunk, sr)
                    t_mlp = time.time()

                    text = get_text(audio_chunk, sr)
                    t_whisper = time.time()

                    audio_logs.append({
                        "VAD_Time_ms": (t_vad - t_start_audio) * 1000,
                        "MLP_Time_ms": (t_mlp - t_vad) * 1000,
                        "Whisper_Time_ms": (t_whisper - t_mlp) * 1000,
                        "Total_Audio_ms": (t_whisper - t_start_audio) * 1000
                    })

                    server_app.socketio.emit('new_transcript', {'name': name, 'text': text, 'confidence': float(confidence)})

                    # --- LOGIC TRẢ LỜI ---
                    if text and len(text) > 2 and (time.time() - last_response_time > 3):
                        text_lower = text.lower()
                        
                        # Xử lý lệnh tắt (Offline)
                        if "tắt hệ thống" in text_lower:
                            server_app.socketio.emit('ai_response', {'text': "Tạm biệt! Đang tắt hệ thống"})
                            time.sleep(1)
                            handle_stop_server_event() # Gọi hàm tắt từ module
                            save_perf_logs()
                            return
                        
                        # Xử lý hỏi đáp (Gemini)
                        else:
                            server_app.socketio.emit('update_status', {'state': 'thinking'})
                            t_gemini_start = time.time()
                            response = ask_gemini(text)
                            t_gemini_end = time.time()

                            if audio_logs:
                                audio_logs[-1]["Gemini_Latency_ms"] = (t_gemini_end - t_gemini_start) * 1000

                            server_app.socketio.emit('ai_response', {'text': response})
                            server_app.socketio.emit('update_status', {'state': 'listening'})
                            last_response_time = time.time()

                except Exception as e:
                    print(f"Lỗi xử lý mic: {e}")
    except Exception: pass
    finally:
        save_perf_logs()

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
    save_perf_logs()
    handle_stop_server_event()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handle_shutdown)
    
    print("Server: http://127.0.0.1:5000")
    # Chạy server từ module server_app
    server_app.socketio.run(server_app.app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)