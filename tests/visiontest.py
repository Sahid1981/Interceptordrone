

from ultralytics import YOLO
import cv2
import time
import threading
from flask import Flask, Response

# configuration
MODEL_PATH = "best.engine"
CAMERA_INDEX = 0        
FRAME_W, FRAME_H = 1280, 720
HFOV_DEG = 70.0         
CONF_MIN = 0.5          
SMOOTH = 0.3            
LOST_TIMEOUT = 2.0      
HTTP_PORT = 8080
JPEG_QUALITY = 80
RECORD_VIDEO = False    


app = Flask(__name__)
latest_jpeg = None
jpeg_lock = threading.Lock()


def mjpeg_generator():
    while True:
        with jpeg_lock:
            frame = latest_jpeg
        if frame is not None:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + frame + b'\r\n')
        time.sleep(0.03)


@app.route('/')
def stream():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def start_server():
    threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=HTTP_PORT,
                               threaded=True, debug=False,
                               use_reloader=False),
        daemon=True,
    ).start()
    print(f"Striimi kaynnissa: http://<jetsonin-ip>:{HTTP_PORT}")



def compute_errors(frame_shape, results, conf_min=CONF_MIN):
    """Palauttaa (err_x, err_y, size, conf) tai None jos ei havaintoa.

    err_x/err_y normalisoitu valille -1..+1 (0 = kuvan keskella),
    size = laatikon leveys / kuvan leveys (kasvaa lahestyttaessa).
    """
    h, w = frame_shape[:2]
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    best = None
    for b in boxes:
        conf = float(b.conf[0])
        if conf < conf_min:
            continue
        x1, y1, x2, y2 = map(float, b.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if best is None or area > best[0]:
            best = (area, x1, y1, x2, y2, conf)
    if best is None:
        return None

    _, x1, y1, x2, y2, conf = best
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    err_x = (cx - w / 2) / (w / 2)
    err_y = (cy - h / 2) / (h / 2)
    size = (x2 - x1) / w
    return err_x, err_y, size, conf



def main():
    model = YOLO("/home/jetson/Interceptordrone/datasets/balloon.engine")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise SystemExit("Kameraa ei saatu auki (tarkista CAMERA_INDEX)")

    writer = None
    if RECORD_VIDEO:
        writer = cv2.VideoWriter("detections.mp4",
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 20, (FRAME_W, FRAME_H))

    start_server()

    
    err_x_s, err_y_s, size_s = 0.0, 0.0, 0.0
    last_seen = 0.0
    state = "SEARCHING"

    
    fps = 0.0
    t0 = time.time()
    n = 0

    global latest_jpeg
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Kameran luku epaonnistui")
                break

            results = model(frame, verbose=False)
            out = compute_errors(frame.shape, results)
            now = time.time()

            if out is not None:
                err_x, err_y, size, conf = out
                err_x_s = (1 - SMOOTH) * err_x_s + SMOOTH * err_x
                err_y_s = (1 - SMOOTH) * err_y_s + SMOOTH * err_y
                size_s = (1 - SMOOTH) * size_s + SMOOTH * size
                last_seen = now
                state = "TRACKING"
                yaw_err_deg = err_x_s * (HFOV_DEG / 2)
                print(f"[{state}] err_x={err_x_s:+.2f} err_y={err_y_s:+.2f} "
                      f"size={size_s:.3f} yaw={yaw_err_deg:+.1f}deg "
                      f"conf={conf:.2f} fps={fps:.1f}")
            else:
                if last_seen > 0 and now - last_seen > LOST_TIMEOUT:
                    state = "LOST"
                elif last_seen == 0:
                    state = "SEARCHING"
                print(f"[{state}] ei havaintoa (fps={fps:.1f})")

            n += 1
            if n % 30 == 0:
                fps = 30 / (now - t0)
                t0 = now

            annotated = results[0].plot()
            cv2.putText(annotated, f"{state}  FPS {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            ok_enc, jpeg = cv2.imencode(
                '.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok_enc:
                with jpeg_lock:
                    latest_jpeg = jpeg.tobytes()

            if writer is not None:
                writer.write(annotated)

    except KeyboardInterrupt:
        print("\nLopetetaan...")
    finally:
        cap.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
