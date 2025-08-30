from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import threading
import subprocess
from ultralytics import YOLO
import io
import base64
import time
import serial

# ====== SETUP ======
app = Flask(__name__)
CORS(app)
model = YOLO("model/ppewa.pt")

# ====== SERIAL ARDUINO ======
try:
    arduino = serial.Serial('COM4', 9600, timeout=1)
    time.sleep(2)  # รอ Arduino รีเซ็ต
    print("✅ Arduino connected")
except Exception as e:
    print("❌ ไม่สามารถเชื่อมต่อ Arduino:", e)
    arduino = None

def send_to_gate(status: str):
    if arduino and status in ["PASS", "NOT PASS"]:
        try:
            arduino.write((status + '\n').encode())
            print(f"📤 ส่งไปยัง Arduino: {status}")
        except Exception as e:
            print(f"❌ ส่งไม่สำเร็จ: {e}")

# ====== กล้อง & วิเคราะห์ YOLO ======
cap = cv2.VideoCapture(0)
latest_frame = None
violations_list = []
last_violation_time = 0
violation_cooldown = 5  # วินาที

def detect_ppe():
    global latest_frame, violations_list, last_violation_time
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        resized_frame = cv2.resize(frame, (1300, 635))
        results = model(resized_frame)
        annotated_frame = results[0].plot()

        names = results[0].names
        boxes = results[0].boxes
        detected = [names[int(cls)] for cls in boxes.cls]

        is_wearing_vest = "Safety Vest" in detected
        is_wearing_hat = "Hardhat" in detected

        violations = []
        if not is_wearing_vest:
            violations.append("Safety Vest")
        if not is_wearing_hat:
            violations.append("Hardhat")

        status_text = "PASS" if len(violations) == 0 else "NOT PASS"
        color = (0, 255, 0) if status_text == "PASS" else (0, 0, 255)

        # === ส่งสถานะให้ Arduino ===
        send_to_gate(status_text)

        cv2.putText(
            annotated_frame,
            status_text,
            (50, annotated_frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            color,
            6,
            cv2.LINE_AA,
        )

        latest_frame = cv2.imencode(".jpg", annotated_frame)[1].tobytes()

        now = time.time()
        if now - last_violation_time > violation_cooldown:
            last_violation_time = now
            encoded_image = base64.b64encode(latest_frame).decode('utf-8')
            timestamp = time.strftime("%H:%M:%S")
            violations_list.append({
                "timestamp": timestamp,
                "missing": violations,
                "image": encoded_image
            })

# ====== API ======
@app.route("/image")
def image():
    if latest_frame is not None:
        return Response(latest_frame, mimetype="image/jpeg")
    else:
        return "No image", 500

@app.route("/violations")
def get_violations():
    return jsonify(violations_list)

@app.route("/start-server")
def start_server():
    global python_process
    if not python_process:
        python_process = subprocess.Popen(["python", "server.py"])
        return "Server started"
    return "Already running"

@app.route("/stop-server")
def stop_server():
    global python_process
    if python_process:
        python_process.terminate()
        python_process = None
        return "Server stopped"
    return "Not running"

# ====== START THREAD ======
if __name__ == "__main__":
    t = threading.Thread(target=detect_ppe)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000)
