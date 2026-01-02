import time
import sys
import os
import signal
import server_app

def do_cleanup():
    # Truy cập biến qua server_app.ten_bien
    if server_app.stop_event.is_set():
        return
        
    server_app.stop_event.set() # Bật cờ dừng
    
    if server_app.voice_stream:
        try:
            server_app.voice_stream.stop()
            server_app.voice_stream.close()
        except Exception as e:
            print(f"Lỗi khi đóng mic: {e}")
    
    time.sleep(1) 

def handle_shutdown(sig, frame):
    do_cleanup()
    print("Hệ thống đã tắt")
    sys.exit(0)

def handle_stop_server_event():
    print("Hệ thống đã tắt")
    os.kill(os.getpid(), signal.SIGINT)
    