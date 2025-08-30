from ultralytics import YOLO
import cv2

# โหลดโมเดล YOLO ที่เทรนไว้
model = YOLO(r"C:\Users\ADMIN\Downloads\WRG CODE\Project WRG\model\ppewa.pt")

# แสดงคลาสทั้งหมดที่โมเดลรองรับ
print("Classes in model:", model.names)