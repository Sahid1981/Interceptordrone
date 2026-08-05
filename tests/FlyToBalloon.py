
import asyncio
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
from ultralytics import YOLO

from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# configuration 

CONFIG = {
    # Connection
    "system_address": "udpin://0.0.0.0:14550",

    # Flight
    "takeoff_altitude_m": 1.0,

    # Camera / model
    "camera_index": 0,            
    "frame_width": 1280,
    "frame_height": 720,
    "model_path": "/home/jetson/Interceptordrone/datasets/balloon.engine",
    "conf_threshold": 0.5,

    # Debug stream
    "stream_port": 8080,
    "stream_width": 640,          
    "stream_quality": 70,
    "stream_max_fps": 10,

    # Control gains (P-controllers on normalized pixel error, range -1..1)
    "kp_yaw_deg_s": 40.0,        
    "kp_vz_m_s": 0.4,             

    # Limits
    "yaw_rate_max_deg_s": 30.0,
    "vz_max_m_s": 0.1,
    "forward_speed_max_m_s": 0.1, 

    # Approach logic
    "centered_threshold": 0.15,   
    "arrive_bbox_frac": 0.20,     

    # Timeouts
    "detection_stale_s": 0.7,     
    "detection_lost_s": 10.0,     
    "mission_timeout_s": 120.0,    
    "control_rate_hz": 10.0,
}

# stream

class MjpegServer:


    def __init__(self, port=8080, width=640, jpeg_quality=70, max_fps=10):
        self.port = port
        self.width = width
        self.quality = jpeg_quality
        self.min_interval = 1.0 / max_fps
        self._lock = threading.Lock()
        self._frame = None
        self._server = None

    
    def publish(self, frame):
        with self._lock:
            self._frame = frame

    
    def _encode_latest(self):
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w > self.width:
            scale = self.width / w
            frame = cv2.resize(frame, (self.width, int(h * scale)))
        ok, jpg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return jpg.tobytes() if ok else None

    def start(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        t0 = time.monotonic()
                        data = server._encode_latest()
                        if data is not None:
                            self.wfile.write(b"--frame\r\n")
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(data)))
                            self.end_headers()
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                        dt = time.monotonic() - t0
                        if dt < server.min_interval:
                            time.sleep(server.min_interval - dt)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # client closed the browser tab

            def log_message(self, *args):
                pass 

        self._server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        print(f"-- Debug stream at http://0.0.0.0:{self.port}")

    def stop(self):
        if self._server:
            self._server.shutdown()


# detection side 


class BalloonDetector(threading.Thread):

    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.cfg = cfg
        self._lock = threading.Lock()
        self._latest = None         
        self._running = True
        self.ready = threading.Event()
        self.failed = False
        self.stream = MjpegServer(
            port=cfg["stream_port"],
            width=cfg["stream_width"],
            jpeg_quality=cfg["stream_quality"],
            max_fps=cfg["stream_max_fps"],
        )

    def run(self):
        cfg = self.cfg
        try:
            model = YOLO(cfg["model_path"])
            cap = cv2.VideoCapture(cfg["camera_index"], cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["frame_width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["frame_height"])
            if not cap.isOpened():
                raise RuntimeError("Camera failed to open")
            self.stream.start()
        except Exception as e:
            print(f"!! Detector init failed: {e}")
            self.failed = True
            self.ready.set()
            return

        self.ready.set()
        w, h = cfg["frame_width"], cfg["frame_height"]

        while self._running:
            ok, frame = cap.read()
            if not ok:
                continue

            results = model(frame, conf=cfg["conf_threshold"], verbose=False)
            boxes = results[0].boxes

            det = None
            if boxes is not None and len(boxes) > 0:
        
                best = max(boxes,
                           key=lambda b: float(b.xywh[0][2]) * float(b.xywh[0][3]))
                cx, cy, bw, _bh = (float(v) for v in best.xywh[0])

                det = {
                    "x_err": (cx - w / 2) / (w / 2),
                    "y_err": (cy - h / 2) / (h / 2),
                    "bbox_frac": bw / w,
                    "ts": time.monotonic(),
                }
                with self._lock:
                    self._latest = det

            annotated = results[0].plot()   # draws bboxes + confidences
            if det is not None:
                cv2.putText(
                    annotated,
                    f"x_err={det['x_err']:+.2f} y_err={det['y_err']:+.2f} "
                    f"bbox={det['bbox_frac']:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(annotated, "no balloon", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # Crosshair 
            cv2.drawMarker(annotated, (w // 2, h // 2), (255, 255, 0),
                           cv2.MARKER_CROSS, 30, 2)
            self.stream.publish(annotated)

        cap.release()
        self.stream.stop()

    def latest(self):
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False


# helpers 


def clamp(v, limit):
    return max(-limit, min(limit, v))

async def print_status_text(drone):
    async for st in drone.telemetry.status_text():
        print(f"[FC] {st.type}: {st.text}")

async def stop_offboard_and_land(drone):
    try:
        await drone.offboard.stop()
    except OffboardError:
        pass
    try:
        print("Landing...")
        await drone.action.land()
    except ActionError as e:
        print(f"!! Land command rejected: {e}")
        return
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print("-- Landed!")
            break


# approach loop 


async def approach_balloon(drone, detector, cfg):
    """Returns 'arrived', 'lost' or 'timeout'."""
    period = 1.0 / cfg["control_rate_hz"]
    start = time.monotonic()
    last_seen = start

    while True:
        now = time.monotonic()

        if now - start > cfg["mission_timeout_s"]:
            print("-- Mission timeout reached")
            return "timeout"

        det = detector.latest()
        fresh = det is not None and (now - det["ts"]) < cfg["detection_stale_s"]

        if fresh:
            last_seen = det["ts"]

            yaw_rate = clamp(cfg["kp_yaw_deg_s"] * det["x_err"],
                             cfg["yaw_rate_max_deg_s"])
            vz = clamp(cfg["kp_vz_m_s"] * det["y_err"], cfg["vz_max_m_s"])

            centered = abs(det["x_err"]) < cfg["centered_threshold"]
            forward = cfg["forward_speed_max_m_s"] if centered else 0.0

            if det["bbox_frac"] >= cfg["arrive_bbox_frac"]:
                print(f"-- Balloon reached (bbox {det['bbox_frac']:.2f} of frame)")
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                return "arrived"

            print(f"   x_err={det['x_err']:+.2f} y_err={det['y_err']:+.2f} "
                  f"bbox={det['bbox_frac']:.2f} -> fwd={forward:.2f} "
                  f"yaw={yaw_rate:+.1f} vz={vz:+.2f}")
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(forward, 0.0, vz, yaw_rate))

        else:
            if now - last_seen > cfg["detection_lost_s"]:
                print("-- Balloon lost for too long")
                return "lost"
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))

        await asyncio.sleep(period)


# main flight logic

async def run():
    cfg = CONFIG

    
    print("Starting balloon detector...")
    detector = BalloonDetector(cfg)
    detector.start()
    detector.ready.wait(timeout=30)
    if detector.failed or not detector.ready.is_set():
        print("!! Detector failed to start - aborting before flight")
        sys.exit(1)
    print("-- Detector running (check the debug stream before arming!)")

    drone = System()
    await drone.connect(system_address=cfg["system_address"])

    print("Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected")
            break

    status_task = asyncio.ensure_future(print_status_text(drone))

    print("Waiting for GPS lock and home position...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position OK")
            break

    try:
        print("Setting flight mode to Hold (pre-arm safe state)...")
        await drone.action.hold()
        print("-- Mode set to Hold")
    except ActionError as e:
        print(f"!! Failed to set Hold mode: {e}")
        sys.exit(1)

    await asyncio.sleep(1)

    try:
        print("Arming...")
        await drone.action.arm()
    except ActionError as e:
        print(f"!! Arm command rejected: {e}")
        sys.exit(1)

    armed_confirmed = False
    async for is_armed in drone.telemetry.armed():
        if is_armed:
            armed_confirmed = True
            print("-- Armed confirmed via telemetry")
            break
        print("-- Waiting for armed confirmation...")
        await asyncio.sleep(0.5)

    if not armed_confirmed:
        print("!! Arm never confirmed - aborting")
        sys.exit(1)

    try:
        print(f"Setting takeoff altitude to {cfg['takeoff_altitude_m']}m...")
        await drone.action.set_takeoff_altitude(cfg["takeoff_altitude_m"])
        print("Taking off...")
        await drone.action.takeoff()
    except ActionError as e:
        print(f"!! Takeoff failed: {e}")
        sys.exit(1)

    print("Waiting for confirmed takeoff (in_air)...")
    took_off = False
    for _ in range(20):
        async for in_air in drone.telemetry.in_air():
            if in_air:
                took_off = True
                print("-- Confirmed in air")
            break
        if took_off:
            break
        await asyncio.sleep(0.5)

    if not took_off:
        print("!! Never left the ground - aborting, attempting disarm")
        try:
            await drone.action.disarm()
        except ActionError:
            pass
        sys.exit(1)

    print("Stabilizing at altitude for 3 seconds...")
    await asyncio.sleep(3)


    try:
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        print("Starting offboard mode...")
        await drone.offboard.start()
        print("-- Offboard active")
    except OffboardError as e:
        print(f"!! Offboard start failed: {e._result.result} - landing")
        await stop_offboard_and_land(drone)
        detector.stop()
        sys.exit(1)

    try:
        result = await approach_balloon(drone, detector, cfg)
        print(f"-- Approach finished: {result}")
        if result == "arrived":
            print("Hovering at balloon for 3 seconds...")
            for _ in range(int(3 * cfg["control_rate_hz"])):
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                await asyncio.sleep(1.0 / cfg["control_rate_hz"])
    finally:
        await stop_offboard_and_land(drone)
        detector.stop()

    try:
        print("Disarming...")
        await drone.action.disarm()
        print("-- Disarmed")
    except ActionError as e:
        print(f"-- Disarm note: {e} (may already be disarmed automatically)")

    status_task.cancel()

    print("Done.")


if __name__ == "__main__":
    asyncio.run(run())
