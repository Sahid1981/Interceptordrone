""" FlyToBalloon version 2.2
    Added Search function and fixed some errors and bugs"""

import asyncio
import sys
import time
import threading
import os
from datetime import datetime
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
    "min_altitude_m": 1.0,

    # Camera / model
    "camera_index": 0,            
    "frame_width": 1280,
    "frame_height": 720,
    "model_path": "/home/jetson/Interceptordrone/datasets/balloon.engine",
    "conf_threshold": 0.35,

    # Debug stream
    "stream_port": 8080,
    "stream_width": 640,          
    "stream_quality": 70,
    "stream_max_fps": 10,
    "recording_dir" : "/home/jetson/Interceptordrone/records/",
    "recording_fps" : 20,

    # Control gains (P-controllers on normalized pixel error, range -1..1)
    "kp_yaw_deg_s": 40.0,        
    "kp_vz_m_s": 0.4,             

    # Limits
    "yaw_rate_max_deg_s": 30.0,
    "vz_max_m_s": 0.2,
    "forward_speed_max_m_s": 0.0, 

    # Approach logic
    "centered_threshold": 0.15,   
    "arrive_bbox_frac": 0.20, 

    # Search behaviour
    "search_yaw_rate_deg_s": 20.0,
    "detect_confirm_frames": 3,
    "settle_cycles": 3,
    "search_timeout_s": 45.0,
    "search_sweep_deg": 360.0,   

    # Timeouts
    "detection_stale_s": 0.7,     
    "track_lost_s": 1.5,     
    "mission_timeout_s": 120.0,    
    "control_rate_hz": 10.0,
    "telemetry_timeout_s": 30.0,
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

        self._server = ThreadingHTTPServer(("10.42.0.1", self.port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        print(f"-- Debug stream at http://10.42.0.1:{self.port}")

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
        self._streak = 0         
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

        self.writer = None
        try:
            os.makedirs(cfg["recording_dir"], exist_ok=True)
            timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
            rec_path = os.path.join(cfg["recording_dir"], f"video_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(rec_path, fourcc, cfg["recording_fps"], (w, h))
            if not self.writer.isOpened():
                print("Video writer failed. Recording disapled")
                self.writer = None
            else:
                print(f"Recording to {rec_path}")
        except Exception as e:
            print(f"Video record failed: {e}")
            self.writer = None

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
                self._streak += 1

                det = {
                    "x_err": (cx - w / 2) / (w / 2),
                    "y_err": (cy - h / 2) / (h / 2),
                    "bbox_frac": bw / w,
                    "ts": time.monotonic(),
                    "streak": self._streak,
                }
                with self._lock:
                    self._latest = det

            else:
                self._streak = 0

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
            if self.writer is not None:
                self.writer.write(annotated)

        cap.release()
        self.stream.stop()
        if self.writer is not None:
            self.writer.release()
            print("Recording saved")

    def latest(self):
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False

# Telemetry cache

class TelemetryCache:

    def __init__(self):
        self.rel_alt_m = None
        self.yaw_deg = None
        self._task = []

    def start(self, drone):
        self._task = [
            asyncio.ensure_future(self._read_position(drone)),
            asyncio.ensure_future(self._read_attitude(drone)),
        ]

    async def _read_position(self,drone):
        async for p in drone.telemetry.position():
            self.rel_alt_m = p.relative_altitude_m

    async def _read_attitude(self, drone):
        async for a in drone.telemetry.attitude_euler():
            self.yaw_deg = a.yaw_deg

    async def wait_ready(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.rel_alt_m is not None and self.yaw_deg is not None:
                return True
            await asyncio.sleep(0.1)
        return False

    def stop(self):
        for t in self._task:
            t.cancel()


# helpers 


def clamp(v, limit):
    return max(-limit, min(limit, v))

def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0

async def print_status_text(drone):
    async for st in drone.telemetry.status_text():
        print(f"[FC] {st.type}: {st.text}")

async def stop_offboard_and_land(drone, timeout=90.0):
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

    async def _wait_landed():    
        async for in_air in drone.telemetry.in_air():
            if not in_air:
                return
    try:
        await asyncio.wait_for(_wait_landed(), timeout)
        print("Landed")
    except asyncio.TimeoutError:
        print("Landing time out")

async def is_airborne(drone, timeout=5.0):
    async def _read():
        async for in_air in drone.telemetry.in_air():
            return in_air
    try:
        return await asyncio.wait_for(_read(), timeout)
    except asyncio.TimeoutError:
        return None

async def safe_disarm(drone):

    airborne = await is_airborne(drone)
    if airborne is None:
        print("May be in flight. Can't disarm")
        return
    if airborne:
        print("Still in air. Can't disarm")
        return
    try:
        await drone.action.disarm()
        print("Disarmed")
    except ActionError as e:
        print(f"Disarm note: {e} (already disarmed?)")


# approach loop 


async def approach_balloon(drone, detector, tel, cfg):
    """Added SEARCH, TRACK, ARRIVED
    Returns 'arrived', 'lost' or 'timeout'."""
    period = 1.0 / cfg["control_rate_hz"]
    start = time.monotonic()
   
    state = "SEARCH"
    search_dir = 1.0
    search_started = start
    swept_deg = 0.0
    prev_yaw = tel.yaw_deg
    settle_left = 0
    last_seen = start

    async def send(fwd, vz, yaw_rate):
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(fwd, 0.0, vz, yaw_rate))
    print(f"SEARCH: scanning {'right' if search_dir > 0 else 'left'}")

    while True:
        now = time.monotonic()

        if now - start > cfg["mission_timeout_s"]:
            print("-- Mission timeout reached")
            await send(0.0, 0.0, 0.0)
            return "timeout"

        det = detector.latest()
        fresh = det is not None and (now - det["ts"]) < cfg["detection_stale_s"]

        # SEARCH

        if state == "SEARCH":
            if tel.yaw_deg is not None and prev_yaw is not None:
                swept_deg += abs(wrap180(tel.yaw_deg - prev_yaw))
            prev_yaw = tel.yaw_deg

            confirmed = fresh and det["streak"] >= cfg["detect_confirm_frames"]
            if confirmed:
                print(f"Balloon confirmed after {det['streak']} frames, settling before track")
                state="SETTLE"
                settle_left = cfg["settle_cycles"]
                await send(0.0, 0.0, 0.0)
                await asyncio.sleep(period)
                continue

            if swept_deg >= cfg["search_sweep_deg"]:
                print("Full sweep. No balloon found.")
                await send(0.0, 0.0, 0.0)
                return "lost"

            if now - search_started > cfg["search_timeout_s"]:
                print("Search timeout")
                await send(0.0, 0.0, 0.0)
                return "lost"

            await send(0.0, 0.0, search_dir * cfg["search_yaw_rate_deg_s"])

        # SETTLE

        elif state == "SETTLE":
            await send(0.0, 0.0, 0.0)
            settle_left -= 1
            if settle_left <= 0:
                state = "TRACK"
                last_seen = now
                print("TRACK")

        # TRACK

        elif state == "TRACK":
            if fresh:
                last_seen = det["ts"]

                if det["bbox_frac"] >= cfg["arrive_bbox_frac"]:
                    print(f"balloon reached (bbox {det['bbox_frac']:.2f} of frame)")
                    await send(0.0, 0.0, 0.0)
                    return "arrived"

                yaw_rate = clamp(cfg["kp_yaw_deg_s"] * det["x_err"],
                             cfg["yaw_rate_max_deg_s"])
                vz = clamp(cfg["kp_vz_m_s"] * det["y_err"], cfg["vz_max_m_s"])

                if (tel.rel_alt_m is not None and tel.rel_alt_m <= cfg["min_altitude_m"] and vz > 0):
                    vz = 0.0

                centered = abs(det["x_err"]) < cfg["centered_threshold"]
                forward = cfg["forward_speed_max_m_s"] if centered else 0.0

                alt_txt = ("--" if tel.rel_alt_m is None else f"{tel.rel_alt_m:.1f}")
                print(f"   x_err={det['x_err']:+.2f} y_err={det['y_err']:+.2f} "
                    f"bbox={det['bbox_frac']:.2f} -> fwd={forward:.2f} "
                    f"yaw={yaw_rate:+.1f} vz={vz:+.2f}")
                await send(forward, vz, yaw_rate)

            elif now - last_seen > cfg["track_lost_s"]: 
                print("Balloon lost, back to SEARCH")
                state = "SEARCH"
                search_started = now
                swept_deg = 0.0
                prev_yaw = tel.yaw_deg
                await send(0.0, 0.0, 0.0)
                
            else:
                await send(0.0, 0.0, 0.0)

        await asyncio.sleep(period)


# main flight logic

async def run():
    cfg = CONFIG

    
    print("Starting balloon detector...")
    detector = BalloonDetector(cfg)
    detector.start()
    detector.ready.wait(timeout=60)
    if detector.failed or not detector.ready.is_set():
        print("!! Detector failed to start - aborting before flight")
        detector.stop()
        sys.exit(1)
    print("-- Detector running (check the debug stream before arming!)")

    drone = System()
    status_task = None
    tel = None
    try:
        await drone.connect(system_address=cfg["system_address"])

        print("Waiting for drone connection...")

        async def _wait_connected():
            async for state in drone.core.connection_state():
                if state.is_connected:
                    return
        try:
            await asyncio.wait_for(_wait_connected(), cfg["telemetry_timeout_s"])
            print("Connected")
        except asyncio.TimeoutError:
            print("No connection")
            sys.exit(1)

        status_task = asyncio.ensure_future(print_status_text(drone))

        tel = TelemetryCache()
        tel.start(drone)

        print("Waiting for GPS lock and home position...")

        async def _wait_health():
            async for health in drone.telemetry.health():
                if health.is_global_position_ok and health.is_home_position_ok:
                    return
        try:
            await asyncio.wait_for(_wait_health(), 120.0)
            print("Global position OK")
        except asyncio.TimeoutError:
            print("No global position")
            sys.exit(1)

        if not await tel.wait_ready():
            print("No altitude/attitude telemetry")
            sys.exit(1)

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

        async def _wait_armed():
            async for is_armed in drone.telemetry.armed():
                if is_armed:
                    return
        try:
            await asyncio.wait_for(_wait_armed(), 15.0)
            print("-- Armed confirmed via telemetry")
        except asyncio.TimeoutError:
            print("Arm never confirmed")
            await safe_disarm(drone)
            sys.exit(1)

        try:
            print(f"Setting takeoff altitude to {cfg['takeoff_altitude_m']}m...")
            await drone.action.set_takeoff_altitude(cfg["takeoff_altitude_m"])
            print("Taking off...")
            await drone.action.takeoff()
        except ActionError as e:
            print(f"!! Takeoff failed: {e}")
            await safe_disarm(drone)
            sys.exit(1)

        print("Waiting to reach takeoff altitude")
        target = 0.9 * cfg["takeoff_altitude_m"]
        deadline = time.monotonic() + 30.0
        reached = False
        while time.monotonic() < deadline:
            if tel.rel_alt_m is not None and tel.rel_alt_m >= target:
                reached = True
                break
            await asyncio.sleep(0.2)

        if not reached:
            print("Never reached takeoff altitude")
            await stop_offboard_and_land(drone)
            sys.exit(1)

        print(f"At {tel.rel_alt_m:.1f}m, stabilizing for 3 seconds")
        await asyncio.sleep(3)


        try:
            await drone.offboard.set_velocity_body( VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            print("Starting offboard mode...")
            await drone.offboard.start()
            print("-- Offboard active")
        except OffboardError as e:
            print(f"!! Offboard start failed: {e._result.result} - landing")
            await stop_offboard_and_land(drone)
            await safe_disarm(drone)
            sys.exit(1)

        
        result = await approach_balloon(drone, detector, tel, cfg)
        print(f"-- Approach finished: {result}")
        if result == "arrived":
            print("Hovering at balloon for 3 seconds...")
            for _ in range(int(3 * cfg["control_rate_hz"])):
                await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                await asyncio.sleep(1.0 / cfg["control_rate_hz"])
    except ActionError as e:
        print(f"Flight action rejected: {e}")
    except OffboardError as e:
        print(f"Offboard error: {e._result.result}")
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("Interrupted (Ctrl-C) - landing, disarming and closing recording...")
    finally:
        await stop_offboard_and_land(drone)
        await safe_disarm(drone)
        detector.stop()
        detector.join(timeout=5.0)
        if detector.is_alive():
            print("Detector failed - recording may be incomplete")
        if tel is not None:
            tel.stop()
        if status_task is not None:
            status_task.cancel()

        tel.stop()
        status_task.cancel()

    print("Done.")


if __name__ == "__main__":
    asyncio.run(run())
