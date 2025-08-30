from ultralytics import YOLO
import cv2

# โหลดโมเดล YOLO ที่เทรนไว้
model = YOLO("model/ppewa.pt")
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (1300, 710))
    results = model(resized_frame)
    annotated_frame = results[0].plot()

    # ดึง label ที่ตรวจจับได้
    names = results[0].names
    boxes = results[0].boxes
    detected = [names[int(cls)] for cls in boxes.cls]

    # ตรวจสอบ PPE เฉพาะ safetyvest และ hardhat
    safetyvest_count = detected.count("Safety Vest")
    hardhat_count = detected.count("Hardhat")

    # กำหนดจำนวนขั้นต่ำที่ต้องการ
    min_safetyvest = 0.7
    min_hardhat = 0.7

    # เงื่อนไข PASS/NOT PASS
    if safetyvest_count >= min_safetyvest and hardhat_count >= min_hardhat:
        status_text = "PASS"
        color = (0, 255, 0)
    else:
        status_text = "Not PASS"
        color = (0, 0, 255)

    # แสดงผลบนภาพ
    cv2.putText(
        annotated_frame,
        status_text,
        (50, annotated_frame.shape[0] - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        color,
        8,
        cv2.LINE_AA,
    )

    cv2.imshow("PPE Detection", annotated_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()