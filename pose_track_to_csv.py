# pose_track_to_csv.py
from ultralytics import YOLO
import cv2
import csv
from collections import defaultdict

# YOLOv8 Pose (Nano) - CPUでOK
model = YOLO("yolov8n-pose.pt")

# 入力動画
SOURCE = "squat8n.mp4"  # カメラなら 0

# 出力CSV
OUT_CSV = "keypoints_tracks.csv"

# COCOの17キーポイント名（YOLOv8 pose準拠）
KP_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
    "left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

# 推論（stream=True で逐次処理）
results_gen = model.track(source=SOURCE, stream=True, tracker="bytetrack.yaml", verbose=False, persist=True)

# CSVヘッダを作成
header = ["frame","track_id"]
for name in KP_NAMES:
    header += [f"{name}_x", f"{name}_y", f"{name}_conf"]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    frame_idx = 0
    for r in results_gen:
        # r.keypoints.xy: [num_person, 17, 2], r.keypoints.conf: [num_person, 17]
        kps = getattr(r, "keypoints", None)
        ids = None
        if hasattr(r, "boxes") and r.boxes is not None and r.boxes.id is not None:
            # track ID は r.boxes.id (tensor) に入る
            ids = r.boxes.id.cpu().numpy().astype(int)

        if kps is None or kps.xy is None or ids is None:
            frame_idx += 1
            continue

        xy = kps.xy.cpu().numpy()         # (N,17,2)
        conf = (kps.conf.cpu().numpy()    # (N,17)
                if kps.conf is not None else None)

        for i, tid in enumerate(ids):
            row = [frame_idx, int(tid)]
            for j in range(len(KP_NAMES)):
                x, y = xy[i, j, 0], xy[i, j, 1]
                c = conf[i, j] if conf is not None else 1.0
                row += [float(x), float(y), float(c)]
            writer.writerow(row)

        frame_idx += 1

print(f"Saved -> {OUT_CSV}")
