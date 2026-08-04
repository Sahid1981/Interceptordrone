from ultralytics import YOLO
//here you put your model path
model = YOLO("")
model.export(format="onnx")