from ultralytics import YOLO
import cv2

# เปิดไฟล์วิดีโอแทนการเปิดกล้อง
video_path = r"C:\Users\ADMIN\Downloads\WRG CODE\Project WRG\test.mp4"  # เปลี่ยนเป็น path ของคลิปที่ต้องการ
cap = cv2.VideoCapture(video_path)

# โหลดโมเดล YOLO
model = YOLO(r"C:\Users\ADMIN\Downloads\WRG CODE\Project WRG\bestnew.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize เฟรมเพื่อลดเวลาในการประมวลผล
    resized_frame = cv2.resize(frame, (640, 640))

    # ตรวจจับวัตถุ
    results = model(resized_frame)

    # แสดงผลลัพธ์
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Detection", annotated_frame)

    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# ปิดไฟล์วิดีโอและหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()
