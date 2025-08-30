from ultralytics import YOLO
import cv2

# Load a pretrained YOLO11n model
# Open the webcam
cap = cv2.VideoCapture(1)

# Load a pretrained YOLO model
model = YOLO(r"C:\Users\ADMIN\Downloads\WRG CODE\Project WRG\model\ppewa.pt")  # Change to your model path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to reduce processing time
    resized_frame = cv2.resize(frame, (640, 640))

    # Perform object detection on the resized frame
    results = model(resized_frame)

    # Display results
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows() 
# Close all OpenCV windows
