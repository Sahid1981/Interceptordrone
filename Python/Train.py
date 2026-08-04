from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="C:\\Users\\sipil\\Downloads\\balloon.v2i.yolov8\\data.yaml", epochs=300, imgsz=640)