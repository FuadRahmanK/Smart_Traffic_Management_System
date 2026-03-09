from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from ultralytics import YOLO
import cv2
import os
import threading
import time

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = YOLO("yolov8n.pt")

yolo_lock = threading.Lock()
traffic_lock = threading.Lock()

caps = {}
latest_frames = {}

locks = {1: threading.Lock(), 2: threading.Lock(), 3: threading.Lock(), 4: threading.Lock()}
stream_tokens = {1: 0, 2: 0, 3: 0, 4: 0}
stream_threads = {1: None, 2: None, 3: None, 4: None}
controller_thread = None

ambulance_present = {1: False, 2: False, 3: False, 4: False}

lane_times = {1: 10, 2: 10, 3: 10, 4: 10}

vehicle_counts = {
    1: {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
    2: {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
    3: {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
    4: {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
}

waiting_score = {1: 0, 2: 0, 3: 0, 4: 0}

active_lane = 1
next_lane = None

lane_start_time = time.time()
lane_green_time = 10

signal_state = "GREEN"

MIN_GREEN = 10
MAX_GREEN = 60
YELLOW_TIME = 3

TARGET_FPS = 24
FRAME_DELAY = 1.0 / TARGET_FPS
DETECT_EVERY_N_FRAMES = 5


def empty_counts():
    return {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}


@app.route("/")
def upload_page():
    return render_template("up.html")


@app.route("/upload", methods=["POST"])
def upload():
    global controller_thread

    for i in range(1, 5):

        file = request.files.get(f"lane{i}")

        if file and file.filename != "":
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            with traffic_lock:

                old_cap = caps.get(i)
                if old_cap:
                    old_cap.release()

                cap = cv2.VideoCapture(path)
                caps[i] = cap

                stream_tokens[i] += 1
                token = stream_tokens[i]

                stream_threads[i] = threading.Thread(
                    target=video_stream, args=(i, token), daemon=True
                )

                latest_frames.pop(i, None)
                vehicle_counts[i] = empty_counts()
                ambulance_present[i] = False

            stream_threads[i].start()

    with traffic_lock:
        if controller_thread is None or not controller_thread.is_alive():
            controller_thread = threading.Thread(
                target=traffic_controller, daemon=True
            )
            controller_thread.start()

    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/signal_status")
def signal_status():

    with traffic_lock:

        elapsed = time.time() - lane_start_time
        remaining = lane_green_time - elapsed

        if remaining < 0:
            remaining = 0

        return jsonify({
            "active_lane": active_lane,
            "next_lane": next_lane,
            "signal_state": signal_state,
            "remaining": int(remaining)
        })


def traffic_controller():

    global active_lane, next_lane, lane_green_time, lane_start_time, signal_state

    while True:

        with traffic_lock:
            ambulance_snapshot = dict(ambulance_present)
            waiting_snapshot = dict(waiting_score)
            lane_times_snapshot = dict(lane_times)

        if any(ambulance_snapshot.values()):

            emergency_lane = next(l for l, v in ambulance_snapshot.items() if v)

            with traffic_lock:
                active_lane = emergency_lane
                signal_state = "GREEN"
                lane_start_time = time.time()

            while True:
                with traffic_lock:
                    if not ambulance_present[emergency_lane]:
                        break
                time.sleep(1)

            continue

        max_wait = max(waiting_snapshot.values())
        candidates = [l for l, v in waiting_snapshot.items() if v == max_wait]

        if len(candidates) == 1:
            chosen_lane = candidates[0]
        else:
            chosen_lane = max(candidates, key=lambda l: lane_times_snapshot[l])

        with traffic_lock:
            next_lane = chosen_lane
            signal_state = "YELLOW"

        time.sleep(YELLOW_TIME)

        with traffic_lock:

            active_lane = next_lane
            signal_state = "GREEN"

            lane_green_time = lane_times[active_lane]
            lane_green_time = max(MIN_GREEN, min(lane_green_time, MAX_GREEN))

            lane_start_time = time.time()

            print(f"\n>>>> GREEN SIGNAL → Lane {active_lane} for {lane_green_time} seconds\n")

            for lane in waiting_score:
                if lane == active_lane:
                    waiting_score[lane] = 0
                else:
                    waiting_score[lane] += 1

        time.sleep(lane_green_time)


def video_stream(lane_id, stream_token):

    clearance_time = {
        "motorcycle": 2,
        "car": 3,
        "bus": 4.5,
        "truck": 5
    }

    frame_index = 0

    while True:

        with traffic_lock:
            cap = caps.get(lane_id)
            current_token = stream_tokens.get(lane_id)

        if cap is None or current_token != stream_token:
            break

        success, frame = cap.read()

        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (640, 480))
        frame_index += 1

        should_detect = frame_index % DETECT_EVERY_N_FRAMES == 0

        if should_detect:

            frame_height = frame.shape[0]

            with yolo_lock:
                results = model(frame, conf=0.4)[0]

            counts = empty_counts()
            ambulance_detected = False
            weighted_time = 0

            for box in results.boxes:

                cls = int(box.cls)
                label = model.names[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                y_center = (y1 + y2) / 2

                # UPDATED DISTANCE FACTOR
                distance_factor = 0.5 + (1 - (y_center / frame_height))

                if label in counts:
                    counts[label] += 1

                if label == "ambulance":
                    ambulance_detected = True

                if label in clearance_time:
                    weighted_time += clearance_time[label] * distance_factor

            raw_time = weighted_time
            total_time = int(max(MIN_GREEN, min(raw_time, MAX_GREEN)))

            with traffic_lock:
                vehicle_counts[lane_id] = counts
                lane_times[lane_id] = total_time
                ambulance_present[lane_id] = ambulance_detected

            print("\n==============================")
            print(f"Lane {lane_id} Traffic Analysis")
            print(f"Weighted Green Time = {raw_time:.2f}")
            print(f"Final Green Time = {total_time}s")
            print("==============================\n")

        cv2.putText(frame, f"Lane {lane_id}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        _, buffer = cv2.imencode(".jpg", frame)

        with locks[lane_id]:
            latest_frames[lane_id] = buffer

        time.sleep(FRAME_DELAY)


@app.route("/video_feed/<int:lane_id>")
def video_feed(lane_id):

    def generate():

        while True:

            if lane_id not in latest_frames:
                time.sleep(0.05)
                continue

            with locks[lane_id]:
                frame = latest_frames[lane_id]

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   frame.tobytes() +
                   b"\r\n")

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False)