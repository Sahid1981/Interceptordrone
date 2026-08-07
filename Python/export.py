from ultralytics import YOLO
//here you put your model path
model = YOLO("/path/to/your/model.pt")
model.export(format="onnx")