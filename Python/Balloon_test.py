from ultralytics import YOLO

model = YOLO(r"C:\Users\sipil\Desktop\Droneprojekti\Python\runs\detect\train\weights\best.pt")
model.predict(source=0, show=True)