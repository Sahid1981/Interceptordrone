# Interceptordrone

An autonomous quadcopter that finds a balloon with computer vision and flies to it.

Built as a summer project at Oamk (Oulu University of Applied Sciences) in cooperation
with Oamk Dronelab. The copter takes off, sweeps the area looking for a red balloon,
locks onto it, flies towards it and lands next to it — without pilot input.

<img width="1494" height="957" alt="image" src="https://github.com/user-attachments/assets/bb3b382d-6b2e-4e39-badd-c2786821e6cb" />

## How it works

The intelligence is split between two computers in the same airframe.

The **flight controller** (ELARION F405 running ArduCopter) handles the physics: it reads
the sensors, keeps the copter stable and turns high-level commands into motor outputs.

The **companion computer** (NVIDIA Jetson Orin Nano Super) does the heavy lifting. It
reads the camera, runs a YOLO model to find the balloon in the frame, and decides where
the copter should go next. The two talk over MAVLink via a serial link.

The control loop is a simple state machine:

| State | What happens |
|---|---|
| `SEARCH` | Yaw slowly in place until a balloon is detected in N consecutive frames |
| `SETTLE` | Hold still for a few cycles so the detection stabilises |
| `TRACK` | Yaw to centre the target, adjust altitude, move forward until the bounding box fills a set fraction of the frame |

Steering is a pair of P-controllers on the normalised pixel error of the detection
(`x_err`, `y_err`), sent to the flight controller as offboard body-frame velocity
commands through MAVSDK.

## Hardware

| Part | Model |
|---|---|
| Frame | AOS UL7 V5 |
| Motors | iFlight XING 2806.5 1300KV |
| Propellers | Gemfan Flash 7040 (7", 3-blade) |
| Flight controller | ELARION F405 (FC + ESC), ArduCopter 4.7.0-dev |
| Companion computer | NVIDIA Jetson Orin Nano Super 8 GB, JetPack 6 |
| Camera (vision) | Arducam B0497C (IMX678, USB3) |
| Camera (FPV) | Foxeer Predator V5 Micro, analog |
| GPS / compass | Sequre M10-252G + QMC5883L |
| Power module | Matek PM20S-2 (dual BEC) |
| Telemetry | MicoAir LR868-F LoRa |
| RC link | ExpressLRS (Radiomaster Boxer / EdgeTX) |
| Battery | Racepow Li-ion 4S2P 8000 mAh (4S1P 5000 mAh for short test flights) |
| Storage | Samsung 990 PRO 1 TB NVMe |

The Jetson connects to the flight controller over UART (`/dev/ttyTHS1`, 921600 baud).
`mavlink-router` fans that single link out to several endpoints so that Mission Planner
and the flight script can both be connected at the same time.

## Software

- **Flight control:** MAVSDK (Python), offboard velocity commands over MAVLink
- **Detection:** Ultralytics YOLO, fine-tuned on a ~20 000 image balloon dataset from
  Roboflow, exported to TensorRT for the Jetson
- **Video:** MJPEG debug stream (lighter on the Jetson than the RTSP stream used earlier)
- **Simulation:** ArduPilot SITL + Gazebo, used to shake out logic bugs before real flights

## Repository layout

```
Flycode/
  FlyToBalloon.py     Main autonomous flight script (v2.2)
datasets/
  balloon.pt          Trained YOLO weights
tests/
  heartbeat.py        Minimal MAVLink connectivity check over UART
  testflighup.py      Arm, take off, hover, land — no vision
  visiontest.py       Detection only, with MJPEG stream, no flight
  testFlight.py       Early MAVSDK goto-location test
  requirements.txt
```

## Setup

### 1. Virtual environment

On the Jetson, the venv **must** be created with `--system-site-packages`:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch`, `torchvision` and `tensorrt` come from JetPack and must not be installed from
PyPI — the generic aarch64 wheels have no CUDA support and the TensorRT engine will not
load.

### 2. Build the TensorRT engine

The repository ships the PyTorch weights (`datasets/balloon.pt`) but not the TensorRT
engine, because an engine is compiled for one specific GPU, JetPack and TensorRT version
and is not portable. Build it on the Jetson itself:

```bash
yolo export model=datasets/balloon.pt format=engine device=0 half=True
```

This produces `datasets/balloon.engine`, which is what the flight script loads.

### 3. Check the configuration

The settings live in the `CONFIG` dictionary at the top of `Flycode/FlyToBalloon.py`.
At minimum, check these before the first run:

| Key | Note |
|---|---|
| `model_path` | Absolute path to `balloon.engine` |
| `camera_index` | V4L2 index of the USB camera |
| `system_address` | Must match the mavlink-router endpoint |
| `recording_dir` | Where annotated flight video is written |

The MJPEG server binds to `10.42.0.1` (the Jetson's own network interface). Change the
address in `MjpegServer.start()` if your network differs, or the server will fail to bind.

## Running

```bash
source .venv/bin/activate
python Flycode/FlyToBalloon.py
```

The script starts the detector first and waits for it before touching the flight
controller. Open `http://10.42.0.1:8080` and confirm the detector actually sees the
balloon **before arming**. Annotated video is recorded to `recording_dir` for later review.

`tmux` is recommended if you run this over SSH, so the flight does not die with the
connection.

## Safety

This is student project code. It arms motors and flies a 7-inch quadcopter autonomously.
Read this section.

- **Bench-test with the propellers removed first.** Verify arming, mode changes and the
  detection loop on the ground before anything spins.
- **Keep a hand on the transmitter at all times.** Switching out of GUIDED returns control
  to the pilot immediately, and this is the primary abort path.
- The following failsafes are configured in ArduPilot and should be verified after any
  parameter change: return-to-launch on RC loss, battery failsafe, geofence, and a kill
  switch on the transmitter.
- **Calibrate the compass properly, outdoors, over USB.** The QMC5883L sits close to the
  ESC and power wiring, and a bad calibration causes circling ("toilet-bowling") in every
  GPS-dependent mode. That directly corrupts the vision correction loop, because the
  copter is no longer pointing where the script thinks it is.
- Fly only where you are legally allowed to. In Finland this means registering as a drone
  operator with Traficom and following the open category rules.

No warranty of any kind. Use at your own risk.

## Known limitations

- The target is assumed **stationary**. There is no motion prediction or lead pursuit.
- Control gains and speed limits are tuned for this specific airframe and payload.
- The TensorRT engine must be rebuilt after any JetPack or TensorRT upgrade.
- Detection quality drops against a bright sky and in low light.
- The MJPEG server address is hardcoded.


## Credits

Written by Juha Jermalainen and Valtteri Sipilä, Information and Communication
Technology, Oamk.

Thanks to for our teacher Lasse Haverinen for the idea, **Oamk Dronelab** — Juha Kyrönlampi, Luka Kyrö, Henry Hinkula, Janne Rajala
and for advice along the way to Leevi Alakörkkö, Iiro Toivari and Timi Lehto.

Built on the work of the ArduPilot, MAVSDK, Ultralytics and Gazebo open source
communities, whose documentation and forums solved more problems on this project than
anything else.

## License

MIT
