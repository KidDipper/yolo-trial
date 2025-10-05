"""
Streamlit app: Detect "HAND UP" with YOLOv8-Pose and emit a CAN-like one-pulse signal + live graph.

How to run
----------
1) Install deps (Windows OK):
   pip install ultralytics opencv-python streamlit numpy pandas altair

2) Put a pose model at ./models/yolov8n-pose.pt (Ultralytics will auto-download if missing).

3) Start the app:
   streamlit run app_hand_raise_pulse.py

4) In the sidebar, choose camera (0) or a video file, tweak thresholds, and press "Start".

What happens
------------
- When the right wrist stays higher (smaller Y) than the right shoulder by MARGIN_PX for WINDOW_SECS continuously,
  we assert a "HAND UP" event for the tracked person (ByteTrack ID).
- On each event we generate a one-shot *pulse* on a logical signal channel (baseline 0 → 1 → 0).
  Pulse width is configurable in milliseconds (PULSE_MS).
- The signal is visualized as a step plot (0/1) over the last N seconds. Think of it as a CAN-TX digital line example.
- The annotated video and event log are shown alongside the chart.

Notes
-----
- This is a *CAN-like* digital pulse visualization, not an actual CAN frame encoder/decoder.
- You can export an event CSV from the sidebar.
- For multiple tracked IDs, we trigger on any ID (first that satisfies the condition). You can switch to a specific ID if needed.
"""

import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from ultralytics import YOLO

# ----------------------
# Constants / Keypoint indices (COCO for YOLOv8-pose)
# ----------------------
R_SHOULDER = 6   # right_shoulder
R_WRIST    = 10  # right_wrist

# ----------------------
# Sidebar controls
# ----------------------
st.set_page_config(page_title="HAND UP → CAN-like Pulse", layout="wide")
st.title("✋ HAND UP → CAN-like One-Pulse Visualizer")

with st.sidebar:
    st.header("Settings")
    source_type = st.selectbox("Video Source", ["Webcam (0)", "Video file"], index=0)
    video_file = None
    if source_type == "Video file":
        video_file = st.text_input("Path to video", "./sources/squat8n.mp4")
    cam_index = 0
    if source_type == "Webcam (0)":
        cam_index = st.number_input("Camera index", value=0, step=1)

    model_path = st.text_input("Model path", "./models/yolov8n-pose.pt")

    st.subheader("Detection thresholds")
    margin_px   = st.slider("MARGIN_PX (px above shoulder)", 5, 80, 20)
    window_secs = st.slider("WINDOW_SECS (continuous)", 0.1, 2.0, 0.5, 0.1)

    st.subheader("Pulse & chart")
    pulse_ms      = st.slider("PULSE_MS (one-shot width, ms)", 20, 1000, 150)
    chart_window  = st.slider("Chart window (seconds)", 3, 60, 10)
    max_fps_limit = st.slider("Max processing FPS (soft cap)", 5, 60, 30)

    st.subheader("Run")
    start = st.button("▶ Start")
    stop  = st.button("■ Stop")
    export_csv = st.button("Export event CSV")

# ----------------------
# Session state init
# ----------------------
if "run" not in st.session_state:
    st.session_state.run = False
if "event_log" not in st.session_state:
    st.session_state.event_log: List[Tuple[float, int]] = []  # (timestamp, track_id)
if "last_pulse_end" not in st.session_state:
    st.session_state.last_pulse_end = 0.0
if "history" not in st.session_state:
    st.session_state.history: Dict[int, Deque[bool]] = defaultdict(lambda: deque(maxlen=200))
if "need_frames" not in st.session_state:
    st.session_state.need_frames = 5
if "fps_est" not in st.session_state:
    st.session_state.fps_est = 30.0

# Handle buttons
if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

# Export CSV
if export_csv:
    if st.session_state.event_log:
        df = pd.DataFrame(st.session_state.event_log, columns=["timestamp", "track_id"])\
               .assign(ts_iso=lambda d: pd.to_datetime(d["timestamp"], unit="s"))
        csv_path = "./hand_up_events.csv"
        df.to_csv(csv_path, index=False)
        st.success(f"Exported {len(df)} events → {csv_path}")
    else:
        st.info("No events yet.")

# ----------------------
# Layout
# ----------------------
col1, col2 = st.columns([3, 2])
video_area = col1.empty()
info_box   = col1.empty()
chart_area = col2.empty()
recent_box = col2.container()

# ----------------------
# Helper: open video source
# ----------------------
def open_capture():
    if source_type == "Webcam (0)":
        cap = cv2.VideoCapture(int(cam_index))
    else:
        if not os.path.exists(video_file):
            st.error(f"Video file not found: {video_file}")
            return None
        cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        st.error("Failed to open video source.")
        return None
    return cap

# ----------------------
# Helper: build step chart (0/1) for last N seconds
# ----------------------
def render_chart(samples: List[Tuple[float, int]]):
    # samples: list of (timestamp_sec, value0or1)
    if not samples:
        chart_area.info("Signal: waiting for data…")
        return
    df = pd.DataFrame(samples, columns=["t", "sig"])  # already limited to window
    # Convert t to relative seconds for nicer axis (0 at right / now)
    now = time.time()
    df["sec_ago"] = now - df["t"]
    df = df.sort_values("sec_ago")  # left=older, right=newer

    base = alt.Chart(df).mark_line(interpolate="step-after").encode(
        x=alt.X("sec_ago:Q", title="seconds ago"),
        y=alt.Y("sig:Q", title="CAN-like signal (0/1)", scale=alt.Scale(domain=[-0.1, 1.1])),
        tooltip=[alt.Tooltip("sec_ago:Q", format=".2f"), "sig:Q"],
    ).properties(height=280)

    chart_area.altair_chart(base, use_container_width=True)

# ----------------------
# Main loop
# ----------------------
if st.session_state.run:
    # Load model once
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

    cap = open_capture()
    if cap is None:
        st.stop()

    # Estimate FPS
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    st.session_state.fps_est = float(fps)
    st.session_state.need_frames = max(3, int(window_secs * st.session_state.fps_est))

    # For chart: keep recent samples (t, value)
    signal_samples: Deque[Tuple[float, int]] = deque(maxlen=5000)

    # Reset buffers
    st.session_state.history = defaultdict(lambda: deque(maxlen=200))
    st.session_state.last_pulse_end = 0.0

    # Main processing
    prev_time = 0.0
    pulse_width_sec = pulse_ms / 1000.0

    info_box.info(f"FPS≈{st.session_state.fps_est:.1f}, need_frames={st.session_state.need_frames}, margin={margin_px}px")

    while st.session_state.run:
        ok, frame = cap.read()
        if not ok:
            # for files, loop; for camera, break
            if source_type == "Video file":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                st.warning("Camera ended.")
                break

        # Soft throttle to cap FPS
        now = time.time()
        if prev_time and (now - prev_time) < (1.0 / max_fps_limit):
            time.sleep(max(0.0, (1.0 / max_fps_limit) - (now - prev_time)))
        prev_time = time.time()

        event_triggered = False
        any_id = None

        # Inference + tracking
        for r in model.track(frame, stream=True, persist=True, verbose=False):
            im = r.plot()

            if r.keypoints is not None and r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.cpu().numpy().astype(int)            # (N,)
                xy  = r.keypoints.xy.cpu().numpy()                    # (N,17,2)
                boxes_xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes.xyxy is not None else None

                for i, tid in enumerate(ids):
                    shoulder_y = xy[i, R_SHOULDER, 1]
                    wrist_y    = xy[i, R_WRIST, 1]

                    is_up = wrist_y < (shoulder_y - margin_px)
                    st.session_state.history[tid].append(is_up)

                    if len(st.session_state.history[tid]) >= st.session_state.need_frames and \
                       all(list(st.session_state.history[tid])[-st.session_state.need_frames:]):
                        event_triggered = True
                        any_id = tid
                        if boxes_xyxy is not None:
                            x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
                            cv2.putText(
                                im, f"ID {tid}: HAND UP",
                                (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
                            )

            # Show annotated frame
            im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            video_area.image(im_rgb, channels="RGB", caption="Annotated stream")

        # Handle pulse generation (one-shot)
        t_now = time.time()
        if event_triggered:
            # Log unique rising edge only if previous pulse ended
            if t_now >= st.session_state.last_pulse_end:
                st.session_state.event_log.append((t_now, int(any_id) if any_id is not None else -1))
            # Start/extend pulse window
            st.session_state.last_pulse_end = max(st.session_state.last_pulse_end, t_now + pulse_width_sec)

        # Current signal state
        sig = 1 if t_now < st.session_state.last_pulse_end else 0
        signal_samples.append((t_now, sig))

        # Trim samples to chart_window seconds
        cutoff = t_now - chart_window
        while signal_samples and signal_samples[0][0] < cutoff:
            signal_samples.popleft()

        # Render chart
        render_chart(list(signal_samples))

        # Recent events view
        recent_box.subheader("Recent HAND UP events")
        if st.session_state.event_log:
            df_ev = pd.DataFrame(st.session_state.event_log[-10:], columns=["timestamp", "track_id"])\
                     .assign(time=lambda d: pd.to_datetime(d["timestamp"], unit="s"))
            recent_box.table(df_ev[["time", "track_id"]].iloc[::-1])
        else:
            recent_box.info("No events yet.")

    cap.release()
    st.info("Stopped.")
else:
    st.caption("Press ▶ Start to begin processing.")
