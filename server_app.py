from flask import Flask
from flask_socketio import SocketIO
import threading

# 1. Khởi tạo Server
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# 2. Các biến điều khiển toàn cục (Global State)
stop_event = threading.Event()
voice_stream = None   # Lưu luồng Mic ở đây để system_utils có thể đóng nó