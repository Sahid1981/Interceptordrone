from ultralytics import YOLO
import numpy as np
from collections import deque
import cv2



model = YOLO("C:\\Users\\sipil\\Desktop\\Droneprojekti\\best.pt")
area_history = deque(maxlen=30)
Frame_boxes = []
Confidences = []


cap = cv2.VideoCapture(0)

kalman = cv2.KalmanFilter(4, 2, 0)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], dtype=np.float32)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0]], dtype=np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
kalman.statePre = np.zeros((4, 1), dtype=np.float32)
kalman.statePost = np.zeros((4, 1), dtype=np.float32)

while True:
    frames_without_detections = 0
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.6, iou=0.5, max_det=1, device='cpu')

    if len(results[0].boxes) == 0:
        print("No detection")
        frames_without_detections += 1
        ##if frames_without_detections >= 50:
           ## print("No detection for 50 frames")
    else:
        Frame_boxes.append(results[0].boxes.xywh[0].tolist())
        Confidences.append(results[0].boxes.conf[0].item())
    
        width = results[0].boxes.xywh[0][2].item()
        height = results[0].boxes.xywh[0][3].item()
        area = width * height
        print("Area:", area)

        x1, y1, x2, y2 = results[0].boxes.xyxy[0].tolist()
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        area_history.append(area)
        average_area = sum(area_history) / len(area_history)
        print("Average area:", average_area)
        prediction = kalman.predict()

        center_x = results[0].boxes.xywh[0][0].item()
        center_y = results[0].boxes.xywh[0][1].item()
        measurement = np.array([[center_x], [center_y]], dtype=np.float32)

        estimated = kalman.correct(measurement)

        est_x = estimated[0][0]
        est_y = estimated[1][0]
        est_vx = estimated[2][0]
        est_vy = estimated[3][0]

        N = 20
        future_x = est_x + est_vx * N
        future_y = est_y + est_vy * N
        cv2.circle(frame, (int(future_x), int(future_y)), 10, (255, 0, 0), 2)

    cv2.imshow("Balloon detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()