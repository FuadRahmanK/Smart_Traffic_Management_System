from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from ultralytics import YOLO
import cv2
import os
import queue
import threading
import time

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- MODELS ----------------
vehicle_model   = YOLO("yolov8n.pt")
ambulance_model = YOLO("best.pt")


INFER_IMGSZ             = 416
# Warm up both models so the first real inference doesn't pay JIT/CUDA
# startup cost, which would appear as a sudden FPS spike on first detection.
import numpy as np
_warmup_frame = np.zeros((INFER_IMGSZ, INFER_IMGSZ, 3), dtype=np.uint8)
vehicle_model(_warmup_frame,   imgsz=INFER_IMGSZ, verbose=False)
ambulance_model(_warmup_frame, imgsz=INFER_IMGSZ, verbose=False)
del _warmup_frame

# Semaphore: cap concurrent ambulance YOLO calls.
# Normal mode  → 2 slots: allows slight overlap across lanes while preventing
#                          a 4-at-once GPU spike.
# Emergency    → enforced by capture gating: only the emg lane ever receives
#                frames, so this semaphore is naturally 1-slot in that phase.
_ambulance_infer_sem = threading.Semaphore(2)

# ---------------- SETTINGS ----------------

# --- Signal timing ---
MIN_GREEN             = 10     # seconds
MAX_GREEN             = 60     # seconds
# YELLOW_TIME is defined below after INFER settings (depends on context comment)
AMBULANCE_GREEN_TIME  = 15     # seconds
AMBULANCE_CHECK_DELAY = 10     # seconds — blackout window (must be < AMBULANCE_GREEN_TIME)
INITIAL_COMPUTE_TIME  = 5      # seconds — startup vehicle scan

# --- Ambulance detection ---
CONF_THRESHOLD           = 0.85
AMBULANCE_CONFIRM_FRAMES = 3
AMBULANCE_MEMORY_FRAMES  = 40

# --- Inference rates ---
AMBULANCE_INFER_EVERY_N = 5    # every 5th frame  (~6 Hz)
VEHICLE_INFER_EVERY_N   = 3    # every 3rd frame  (~10 Hz) — denser during yellow window


# --- Signal timing (derived) ---
# Yellow window must be long enough for all 4 vehicle YOLO threads to each
# complete multiple inferences.  At ~50 ms/inference and 4 concurrent threads
# a 5 s window gives ~25 inferences per lane — enough for reliable peak count.
YELLOW_TIME = 5      # seconds (was 3 — too short for accurate vehicle scan)

# --- Display rates ---
# Fixed 30 FPS throughout. Emergency is the lightest phase (zero YOLO
# inferences during blackout) so there is no need to reduce frame rate.
TARGET_FPS  = 30
FRAME_DELAY = 1.0 / TARGET_FPS

# --- Encode settings ---
# Quality 80 is ~2-3x faster than default 95 with no visible difference.
JPEG_QUALITY = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
ENCODE_SIZE  = (640, 480)

# How often capture threads re-read shared state from traffic_lock.
# Between checks the thread uses cached values — reduces lock acquisitions
# from 120/s (4 threads × 30 FPS) to ~20/s with no meaningful staleness.
STATE_CACHE_EVERY = 6    # frames

# Clearance time per vehicle type (seconds) used in green-time calculation.
CLEARANCE_TIME = {
    "motorcycle": 2,
    "car":        3,
    "bus":        4.5,
    "truck":      5,
}


# ---------------- GLOBAL STATE ----------------
traffic_lock = threading.Lock()    # guards ALL shared state listed below

caps          = {}
stream_tokens = {i: 0 for i in range(1, 5)}

# Per-lane queues.
# display_queue  : capture → encode    — always fed,          maxsize=2
# vehicle_queue  : capture → vehicle   — compute window only, maxsize=1
# ambulance_queue: capture → ambulance — emergency-gated,     maxsize=1
display_queue   = {i: queue.Queue(maxsize=2) for i in range(1, 5)}
vehicle_queue   = {i: queue.Queue(maxsize=1) for i in range(1, 5)}
ambulance_queue = {i: queue.Queue(maxsize=1) for i in range(1, 5)}

latest_frames = {}
frame_locks   = {i: threading.Lock() for i in range(1, 5)}

# Per-lane overlay locks — finer granularity than a single global lock.
# Encode thread for lane N only acquires overlay_locks[N], so an ambulance
# inference write on lane 1 never stalls display of lanes 2, 3, or 4.
overlay_locks    = {i: threading.Lock() for i in range(1, 5)}
lane_annotations = {i: [] for i in range(1, 5)}

controller_thread = None

# ---- Signal state ----
active_lane     = 1
next_lane       = None
signal_state    = "GREEN"
lane_start_time = time.time()
lane_green_time = MIN_GREEN

lane_times    = {i: 0         for i in range(1, 5)}   # raw peak score, reset each yellow
waiting_score = {i: 0         for i in range(1, 5)}

# compute_active: True only during startup scan + each yellow phase.
# Capture threads gate vehicle_queue behind this flag.
compute_active = False

# ---- Ambulance / emergency state ----
ambulance_counter = {i: 0     for i in range(1, 5)}
ambulance_memory  = {i: 0     for i in range(1, 5)}
ambulance_present = {i: False for i in range(1, 5)}

emergency_mode              = False
emergency_lane_id           = None
emergency_check_resume_time = 0.0


# ---------------- HELPERS ----------------

def _pick_best_lane():
    """
    Call under traffic_lock.
    Highest waiting_score wins. Tie-break: highest lane_times (most vehicles).
    On first run all scores are 0 so the most-congested lane goes first.
    """
    max_score  = max(waiting_score.values())
    candidates = [l for l, v in waiting_score.items() if v == max_score]
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda l: lane_times[l])


def _trigger_emergency(lane_id):
    """
    Call under traffic_lock.
    Overrides signal to lane_id for AMBULANCE_GREEN_TIME.
    Drains all vehicle queues and non-emergency ambulance queues so stale
    frames already in queues don't cause a burst of inferences right at
    the moment of override — that burst was the FPS spike on green override.
    """
    global emergency_mode, emergency_lane_id, emergency_check_resume_time
    global active_lane, next_lane, signal_state, lane_green_time, lane_start_time
    global compute_active

    emergency_mode              = True
    emergency_lane_id           = lane_id
    active_lane                 = lane_id
    next_lane                   = None
    signal_state                = "GREEN"
    lane_green_time             = AMBULANCE_GREEN_TIME
    lane_start_time             = time.time()
    emergency_check_resume_time = time.time() + AMBULANCE_CHECK_DELAY
    waiting_score[lane_id]      = 0
    compute_active              = False

    for i in range(1, 5):
        _drain(vehicle_queue[i])
        if i != lane_id:
            _drain(ambulance_queue[i])


def _clear_emergency(emg_lane):
    """
    Call under traffic_lock.
    Clears emergency state and adjusts waiting scores so the controller's
    normal scheduling cycle resumes without immediately re-picking emg_lane.
    overlay_lock is intentionally NOT acquired here — the caller must clear
    lane_annotations after releasing traffic_lock to avoid holding two locks
    simultaneously (which blocks encode threads for the full clearing window).
    """
    global emergency_mode, emergency_lane_id, emergency_check_resume_time

    emergency_mode              = False
    emergency_lane_id           = None
    emergency_check_resume_time = 0.0

    for i in range(1, 5):
        ambulance_counter[i] = 0
        ambulance_memory[i]  = 0
        ambulance_present[i] = False
        if i != emg_lane:
            waiting_score[i] += 1   # bump priority for lanes that waited
    waiting_score[emg_lane] = 0     # emergency lane goes to back of queue


def _drain(q):
    """Drain a queue without blocking."""
    while True:
        try:   q.get_nowait()
        except queue.Empty: break


# ---------------- ROUTES ----------------

@app.route("/")
def upload_page():
    return render_template("up.html")


@app.route("/upload", methods=["POST"])
def upload():
    global controller_thread, compute_active

    for i in range(1, 5):
        file = request.files.get(f"lane{i}")
        if not (file and file.filename):
            continue

        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)
        cap = cv2.VideoCapture(path)

        with traffic_lock:
            old = caps.get(i)
            if old is not None:
                old.release()
            caps[i] = cap
            stream_tokens[i] += 1
            token = stream_tokens[i]

        for q in (display_queue[i], vehicle_queue[i], ambulance_queue[i]):
            _drain(q)

        threading.Thread(target=_capture_thread,   args=(i, token), daemon=True).start()
        threading.Thread(target=_vehicle_thread,   args=(i, token), daemon=True).start()
        threading.Thread(target=_ambulance_thread, args=(i, token), daemon=True).start()
        threading.Thread(target=_encode_thread,    args=(i, token), daemon=True).start()

    with traffic_lock:
        compute_active = True   # open vehicle gate for startup scan

    if controller_thread is None or not controller_thread.is_alive():
        controller_thread = threading.Thread(target=_traffic_controller, daemon=True)
        controller_thread.start()

    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/signal_status")
def signal_status():
    with traffic_lock:
        elapsed   = time.time() - lane_start_time
        remaining = max(0, lane_green_time - elapsed)
        return jsonify({
            "active_lane":    active_lane,
            "next_lane":      next_lane,
            "signal_state":   signal_state,
            "remaining":      int(remaining),
            "emergency_lane": emergency_lane_id,
        })


# ---------------- TRAFFIC CONTROLLER ----------------
#
# STARTUP
#   compute_active=True set by upload() before this thread starts.
#   Sleep INITIAL_COMPUTE_TIME → vehicle threads scan all 4 lanes.
#   compute_active=False → fall into main loop.
#
# NORMAL CYCLE
#   [emergency watchdog at top of every iteration]
#   pick best lane → YELLOW + compute_active=True → sleep YELLOW_TIME
#   check: did emergency fire during yellow? → if yes, skip green, back to top
#   compute_active=False → GREEN (uses fresh lane_times from yellow scan)
#   idle sleep for green duration, ambulance threads watch in background
#   emergency fires during idle → break → back to top
#
# EMERGENCY
#   watchdog: sleep 0.1s ticks until window expires → force-clear → continue
#
def _traffic_controller():
    global active_lane, next_lane, signal_state
    global lane_start_time, lane_green_time
    global compute_active

    # Startup: wait for initial vehicle scan then close the gate.
    time.sleep(INITIAL_COMPUTE_TIME)
    with traffic_lock:
        compute_active = False

    while True:

        # ---- EMERGENCY WATCHDOG ----
        with traffic_lock:
            if emergency_mode:
                if time.time() >= lane_start_time + lane_green_time:
                    _clear_emergency(emergency_lane_id)
                    # Fall through to normal scheduling immediately.
                else:
                    time.sleep(0.1)
                    continue

        # ---- YELLOW PHASE ----
        with traffic_lock:
            chosen         = _pick_best_lane()
            next_lane      = chosen
            signal_state   = "YELLOW"
            compute_active = True    # open vehicle gate for yellow duration
            # Reset all lane_times to 0 so vehicle threads accumulate a fresh
            # peak score for this yellow window only.  Stale readings from the
            # previous cycle (or MIN_GREEN initialisation) must not pollute the
            # next green-time calculation.
            for i in range(1, 5):
                lane_times[i] = 0
                _drain(vehicle_queue[i])   # discard any leftover frames too

        time.sleep(YELLOW_TIME)      # vehicle threads compute here

        # Check: did emergency fire during yellow?
        # Without this the controller would overwrite active_lane and
        # lane_start_time, destroying the emergency state just set.
        with traffic_lock:
            if emergency_mode:
                compute_active = False
                for i in range(1, 5):
                    _drain(vehicle_queue[i])
                continue             # back to top → watchdog handles it

        # ---- GREEN PHASE ----
        with traffic_lock:
            compute_active  = False
            active_lane     = next_lane
            next_lane       = None
            signal_state    = "GREEN"
            # lane_times[active_lane] holds the raw peak weighted_time from
            # the just-completed yellow window.  Clamp to [MIN_GREEN, MAX_GREEN]
            # here so the vehicle thread never needs to know about those limits.
            lane_green_time = max(MIN_GREEN, min(lane_times[active_lane], MAX_GREEN))
            lane_start_time = time.time()

            for lane in waiting_score:
                if lane == active_lane:
                    waiting_score[lane] = 0
                else:
                    waiting_score[lane] += 1

        # ---- IDLE: hold green until timeout or emergency ----
        deadline = time.time() + lane_green_time
        while time.time() < deadline:
            with traffic_lock:
                if emergency_mode:
                    break
            time.sleep(0.1)


# ---------------- CAPTURE THREAD ----------------
#
# Sole gating point for all inference queues:
#
#   display_queue  → always (smooth MJPEG regardless of mode)
#
#   vehicle_queue  → only when compute_active=True AND not emergency
#                    (startup scan + yellow phases only)
#
#   ambulance_queue:
#     normal mode         → all lanes every AMBULANCE_INFER_EVERY_N frames
#     emergency blackout  → nobody
#     emergency resume    → emergency lane only every AMBULANCE_INFER_EVERY_N
#
# OPTIMISATION 1 — Adaptive FPS:
#   Normal: sleep DELAY_NORMAL (1/30 s).
#   Emergency: sleep DELAY_EMERGENCY (1/15 s).
#   Halves all capture+encode thread work during emergency with no logic impact.
#
# OPTIMISATION 2 — Lock caching:
#   Shared state is re-read from traffic_lock every STATE_CACHE_EVERY frames.
#   Between checks the thread uses cached values, reducing lock acquisitions
#   from 120/s down to ~20/s across all 4 capture threads.
#   stream_token is still checked every iteration to catch re-uploads promptly.
#
def _capture_thread(lane_id, stream_token):
    frame_count = 0

    # Cached state — refreshed every STATE_CACHE_EVERY frames.
    # Crucially, the stream-token check is also batched here so the capture
    # thread only acquires traffic_lock ~5 times/sec instead of 30 times/sec.
    # 4 capture threads × 30fps was 120 lock acquisitions/sec — these competed
    # directly with the emergency ambulance inference thread which also needs
    # traffic_lock.  Batching drops it to ~20/sec total across all 4 threads.
    cached_cap         = None
    cached_in_emg      = False
    cached_is_emg_lane = False
    cached_resume_time = 0.0
    cached_compute     = False

    while True:

        # Batch all traffic_lock reads together every STATE_CACHE_EVERY frames.
        # Token change (re-upload) is detected within ~200 ms — acceptable.
        if frame_count % STATE_CACHE_EVERY == 0:
            with traffic_lock:
                if stream_tokens[lane_id] != stream_token:
                    return
                cached_cap         = caps.get(lane_id)
                cached_in_emg      = emergency_mode
                cached_is_emg_lane = (emergency_lane_id == lane_id)
                cached_resume_time = emergency_check_resume_time
                cached_compute     = compute_active

        if cached_cap is None:
            time.sleep(0.05)
            continue

        ret, frame = cached_cap.read()
        if not ret:
            cached_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (640, 480))
        frame_count += 1

        # 1. Display — always, at full resolution for encode thread.
        try:   display_queue[lane_id].put_nowait((frame.copy(), frame_count))
        except queue.Full: pass

        now = time.time()

        # 2. Vehicle — only during compute window, never during emergency.
        if cached_compute and not cached_in_emg:
            if frame_count % VEHICLE_INFER_EVERY_N == 0:
                try:   vehicle_queue[lane_id].put_nowait(frame.copy())
                except queue.Full: pass

        # 3. Ambulance — gated by emergency state.
        if cached_in_emg:
            if cached_is_emg_lane and now >= cached_resume_time:
                if frame_count % AMBULANCE_INFER_EVERY_N == 0:
                    try:   ambulance_queue[lane_id].put_nowait(frame.copy())
                    except queue.Full: pass
        else:
            if frame_count % AMBULANCE_INFER_EVERY_N == 0:
                try:   ambulance_queue[lane_id].put_nowait(frame.copy())
                except queue.Full: pass

        # Fixed 30 FPS — same rate in all modes.
        time.sleep(FRAME_DELAY)


# ---------------- VEHICLE THREAD ----------------
#
# Purely passive: blocks on vehicle_queue, runs YOLO, updates lane_times.
# Capture thread guarantees frames only arrive during compute windows.
#
# KEY DESIGN: lane_times stores the RAW peak weighted_time observed across
# ALL frames in the current yellow window (reset to 0 at yellow start).
# Taking the MAX rather than overwriting means a single low-count frame
# (e.g. a momentary occlusion at the end of yellow) cannot erase good
# readings from earlier in the same window.  The green-time clamp to
# [MIN_GREEN, MAX_GREEN] is applied at read time in the controller, not here,
# so lane_times always carries the best raw measurement.
#
# token check is batched every STATE_CACHE_EVERY frames — same pattern as
# capture/encode threads — to avoid acquiring traffic_lock 4×/sec while idle.
#
def _vehicle_thread(lane_id, stream_token):

    token_check = 0

    while True:

        token_check += 1
        if token_check % STATE_CACHE_EVERY == 0:
            with traffic_lock:
                if stream_tokens[lane_id] != stream_token:
                    return

        try:
            frame = vehicle_queue[lane_id].get(timeout=1.0)
        except queue.Empty:
            continue

        frame_height = frame.shape[0]
        results      = vehicle_model(frame, conf=0.4, imgsz=INFER_IMGSZ)[0]
        raw_time     = 0.0

        for box in results.boxes:
            cls   = int(box.cls)
            label = vehicle_model.names[cls]
            if label not in CLEARANCE_TIME:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            y_center        = (y1 + y2) / 2
            distance_factor = 0.5 + (1.0 - (y_center / frame_height))
            raw_time       += CLEARANCE_TIME[label] * distance_factor

        with traffic_lock:
            # Accumulate peak: keep the highest raw score seen this window.
            # Controller resets lane_times[i] = 0 at the start of each yellow
            # phase so this never carries stale data across cycles.
            if raw_time > lane_times[lane_id]:
                lane_times[lane_id] = raw_time


# ---------------- AMBULANCE THREAD ----------------
#
# 4 threads, one per lane, running in parallel (YOLO releases the GIL).
# During emergency blackout: capture feeds nothing → all 4 block on get().
# During emergency resume: capture feeds emergency lane only → 3 remain blocked.
#
# STALE FRAME GUARD: _trigger_emergency drains non-emergency queues but a
# thread may have already called get() and hold a frame before the drain ran.
# This guard discards that frame without inference, preventing spurious clears.
#
def _ambulance_thread(lane_id, stream_token):

    while True:
        with traffic_lock:
            if stream_tokens[lane_id] != stream_token:
                return

        try:
            frame = ambulance_queue[lane_id].get(timeout=1.0)
        except queue.Empty:
            continue

        # Read state after getting the frame — reflects current reality.
        with traffic_lock:
            in_emg   = emergency_mode
            emg_lane = emergency_lane_id

        # Hard gate: during emergency only the emg lane runs inference.
        # Non-emg lanes should receive no frames (capture gating ensures this),
        # but if a stale frame slipped through before the drain, discard it here
        # without running YOLO — keeps 3 idle lanes completely inference-free.
        if in_emg and emg_lane != lane_id:
            continue

        _run_ambulance_inference(frame, lane_id, in_emg)


def _run_ambulance_inference(frame, lane_id, in_emergency):
    """Run one ambulance forward pass and update emergency state."""

    # Semaphore caps concurrent YOLO calls to 2, preventing a 4-thread GPU
    # spike at the exact moment ambulance is first detected in normal mode.
    with _ambulance_infer_sem:
        results = ambulance_model(frame, imgsz=INFER_IMGSZ)[0]

    detected_this_frame = any(
        float(b.conf[0]) >= CONF_THRESHOLD for b in results.boxes
    )
    annotations = [
        (int(b.xyxy[0][0]), int(b.xyxy[0][1]),
         int(b.xyxy[0][2]), int(b.xyxy[0][3]), "AMBULANCE")
        for b in results.boxes if float(b.conf[0]) >= CONF_THRESHOLD
    ]

    with overlay_locks[lane_id]:
        lane_annotations[lane_id] = annotations

    # Track whether _clear_emergency was called so we can clear all overlays
    # AFTER releasing traffic_lock (avoids holding two locks simultaneously).
    should_clear_all_overlays = False

    with traffic_lock:

        if in_emergency:
            # RESUME WINDOW (t = 10–15 s): direct yes/no, memory bypassed.
            if detected_this_frame:
                ambulance_counter[lane_id] = AMBULANCE_CONFIRM_FRAMES
                ambulance_memory[lane_id]  = AMBULANCE_MEMORY_FRAMES
                _trigger_emergency(lane_id)
            else:
                _clear_emergency(lane_id)
                should_clear_all_overlays = True

        else:
            # NORMAL MODE: counter + memory for occlusion robustness.
            if detected_this_frame:
                ambulance_counter[lane_id] += 1
            else:
                ambulance_counter[lane_id] = 0

            if ambulance_counter[lane_id] >= AMBULANCE_CONFIRM_FRAMES:
                ambulance_memory[lane_id] = AMBULANCE_MEMORY_FRAMES

            if ambulance_memory[lane_id] > 0:
                ambulance_detected         = True
                ambulance_present[lane_id] = True
                ambulance_memory[lane_id] -= 1
                if ambulance_memory[lane_id] == 0:
                    ambulance_present[lane_id] = False
            else:
                ambulance_detected         = False
                ambulance_present[lane_id] = False

            if ambulance_detected:
                _trigger_emergency(lane_id)

    # Clear all lane overlays only AFTER releasing traffic_lock so encode
    # threads are never blocked on two locks at the same time.
    # Each overlay_locks[i] is per-lane so these 4 acquires are non-contending.
    if should_clear_all_overlays:
        for i in range(1, 5):
            with overlay_locks[i]:
                lane_annotations[i] = []


# ---------------- ENCODE THREAD ----------------
#
# Composites ambulance overlays onto display frames → JPEG → latest_frames.
# Runs at full 30 FPS and 640×480 in all modes.
# JPEG quality 80 is the only optimisation here — invisible difference,
# ~2-3x faster encode than the default 95.
#
# CRITICAL OPTIMISATION: stream token is now checked every STATE_CACHE_EVERY
# frames instead of every frame.  The old code acquired traffic_lock 120 times/
# sec (4 threads × 30 fps).  When _trigger_emergency or _clear_emergency holds
# that lock, ALL 4 encode threads stall simultaneously — this was the main
# cause of FPS drops on ambulance detection.  Checking every 6 frames reduces
# acquisitions to ~20/sec with zero impact (token only changes on re-upload).
#
def _encode_thread(lane_id, stream_token):

    token_check = 0

    while True:

        # Check stream token infrequently — it only changes on re-upload.
        token_check += 1
        if token_check % STATE_CACHE_EVERY == 0:
            with traffic_lock:
                if stream_tokens[lane_id] != stream_token:
                    return

        try:
            frame, _ = display_queue[lane_id].get(timeout=1.0)
        except queue.Empty:
            continue

        with overlay_locks[lane_id]:
            annotations = list(lane_annotations[lane_id])

        for (x1, y1, x2, y2, label) in annotations:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, f"Lane {lane_id}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Quality 80: ~2-3x faster than default 95, no visible difference.
        ret, buffer = cv2.imencode(".jpg", frame, JPEG_QUALITY)
        if ret:
            with frame_locks[lane_id]:
                latest_frames[lane_id] = buffer


# ---------------- VIDEO FEED ROUTE ----------------

@app.route("/video_feed/<int:lane_id>")
def video_feed(lane_id):

    def generate():
        last_frame = None
        while True:
            with frame_locks[lane_id]:
                frame = latest_frames.get(lane_id)

            if frame is None:
                time.sleep(0.02)
                continue
            if frame is last_frame:
                time.sleep(0.01)
                continue

            last_frame = frame
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame.tobytes()
                + b"\r\n"
            )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------- MAIN ----------------

if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False)