import cv2
from ultralytics import YOLO
import time

# Load the YOLO model (will auto-download the first time)
model = YOLO("yolo11n.pt")
print("✅ Model loaded!")

# Open your webcam (0 = default camera)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🚀 Starting YOLO detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame, conf=0.5, verbose=False)
    
    # Draw bounding boxes
    annotated_frame = results[0].plot()
    
    # Show the frame
    cv2.imshow("YOLO Test - Press 'q' to quit", annotated_frame)
    
    # Print detection info to console
    detections = results[0].boxes
    if len(detections) > 0:
        print(f"🔍 Detected {len(detections)} objects")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Test complete!")