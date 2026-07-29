# Smart Traffic System

An AI-powered adaptive traffic-signal prototype that analyses four road video feeds in real time. It uses YOLO object detection to estimate lane clearance time, dynamically selects the next lane to receive a green signal, and gives priority to detected ambulances.

## Features

- Upload one video feed for each of four roads.
- Detect cars, motorcycles, buses, and trucks with YOLOv8.
- Estimate a lane's required green-light duration from detected vehicles and their position in the frame.
- Allocate green time dynamically between **10** and **60 seconds**.
- Use a fairness score to avoid starving lanes that have waited longer.
- Detect ambulances with a dedicated trained model and switch immediately to emergency priority mode.
- View all four annotated video streams, signal lights, countdowns, and ambulance status in a web dashboard.
- Reset the system and load a new set of feeds without restarting the server.

## How it works

1. The user uploads four road videos through the setup page.
2. The system samples frames from each feed in parallel.
3. During the signal-planning phase, YOLOv8 detects supported vehicle types and calculates an estimated clearance time for each lane.
4. The lane with the highest waiting priority receives the next green signal; its green duration is based on the detected traffic load.
5. Ambulance detection runs continuously. After confirmation across consecutive frames, the corresponding lane receives an emergency green signal for 15 seconds.

## Tech stack

- **Backend:** Flask
- **Computer vision:** Ultralytics YOLOv8, OpenCV
- **Numerical processing:** NumPy
- **Frontend:** HTML, CSS, JavaScript

## Project structure

```text
smart_traffic_system/
|-- combined.py              # Flask app, detection pipeline, and signal controller
|-- yolov8n.pt               # YOLOv8 model for vehicle detection
|-- best.pt                  # Custom ambulance-detection model
|-- templates/
|   |-- up.html              # Four-video upload page
|   `-- dashboard.html       # Live traffic-control dashboard
|-- static/
|   |-- theme.css            # Shared styling
|   `-- uploads/             # Uploaded videos (created/used at runtime)
`-- vids/                    # Sample traffic videos
```

## Requirements

- Python 3.9 or newer
- An OpenCV installation with video-codec support
- The included model files: `yolov8n.pt` and `best.pt`

Install the Python packages:

```bash
pip install flask ultralytics opencv-python numpy
```

> For NVIDIA GPU acceleration, install the PyTorch build appropriate for your CUDA version before installing or running Ultralytics. The application also works on CPU, but inference may be slower.

## Run locally

```bash
git clone https://github.com/FuadRahmanK/smart_traffic_system.git
cd smart_traffic_system
python combined.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. Select four video files - one per road - and choose **Initialise System**.

The repository includes sample `.MOV` files under `vids/` that can be used for testing.

## Configuration

The main settings are near the top of [`combined.py`](combined.py):

| Setting | Default | Purpose |
| --- | ---: | --- |
| `MIN_GREEN` | 10 s | Minimum green-signal duration |
| `MAX_GREEN` | 60 s | Maximum green-signal duration |
| `YELLOW_TIME` | 5 s | Yellow-signal transition duration |
| `AMBULANCE_GREEN_TIME` | 15 s | Emergency green duration |
| `CONF_THRESHOLD` | 0.85 | Minimum ambulance-detection confidence |
| `INFER_IMGSZ` | 416 px | YOLO inference image size |

## Vehicle clearance-time model

The controller estimates lane demand using these base clearance-time weights:

| Vehicle type | Base time |
| --- | ---: |
| Motorcycle | 2 s |
| Car | 3 s |
| Bus | 4.5 s |
| Truck | 5 s |

Each detected vehicle is additionally weighted by its vertical position in the video frame, so vehicles closer to the junction contribute more to the estimated clearance time.

## Notes

- This is a prototype designed for recorded video feeds and demonstration use.
- Uploaded files are saved in `static/uploads/`; do not use untrusted public uploads without adding file validation, size limits, and secure file-name handling.
- The supplied `best.pt` model must be compatible with the installed Ultralytics version.
