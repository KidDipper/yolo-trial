# pose_action_hand_raise.py
# CPU + uv でOK。右手首が右肩より一定ピクセル以上「上」に
# 連続して存在したら「HAND UP」と判定し、画面表示＆ログ出力します。

from ultralytics import YOLO
import cv2
from collections import defaultdict, deque

# ----------------------
# パラメータ
# ----------------------
VIDEO_SOURCE = "squat8n.mp4"   # Webカメラなら 0
VIDEO_SOURCE = 0   # Webカメラなら 0
MODEL_PATH   = "yolov8n-pose.pt"  # 8nのposeモデル（自動DLされます）
MARGIN_PX    = 20            # 肩よりどれだけ上なら「上げた」とみなすか（px）
WINDOW_SECS  = 0.5           # 何秒連続で上にあれば「HAND UP」確定か
HISTORY_MAX  = 200           # 履歴保存の最大フレーム数（必要に応じて調整）

# ----------------------
# 初期化
# ----------------------
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError("動画/カメラを開けません。VIDEO_SOURCE を確認してください。")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
need_frames = max(3, int(WINDOW_SECS * fps))  # 連続判定に必要なフレーム数
print(f"[INFO] FPS={fps:.1f}, need_frames={need_frames}")

# IDごとに直近の「手が肩より上か？」フラグを溜める
history = defaultdict(lambda: deque(maxlen=HISTORY_MAX))
frame_idx = 0

# COCOキーポイントのインデックス（YOLOv8 pose）
R_SHOULDER = 6   # right_shoulder
R_WRIST    = 10  # right_wrist

# ----------------------
# メインループ
# ----------------------
while True:
    ok, frame = cap.read()
    if not ok:
        break

    # 1フレーム分を追跡つきで推論（trackerは省略＝ByteTrack）
    # stream=True で逐次結果を得る
    for r in model.track(frame, stream=True, persist=True, verbose=False):
        im = r.plot()

        if r.keypoints is not None and r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)            # (N,)
            xy  = r.keypoints.xy.cpu().numpy()                    # (N,17,2)
            boxes_xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes.xyxy is not None else None

            for i, tid in enumerate(ids):
                shoulder_y = xy[i, R_SHOULDER, 1]
                wrist_y    = xy[i, R_WRIST, 1]

                # 画像座標は下向きが+：手首が肩より上にあるなら True
                is_up = wrist_y < (shoulder_y - MARGIN_PX)
                history[tid].append(is_up)

                # --- 判定ロジック（ここが「手を挙げた」確定箇所）---
                if len(history[tid]) >= need_frames and all(list(history[tid])[-need_frames:]):
                    # ログ（フレーム番号とIDを印字）
                    print(f"[HAND UP] frame={frame_idx} id={tid}")

                    # 画面にも表示
                    if boxes_xyxy is not None:
                        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
                        cv2.putText(
                            im, f"ID {tid}: HAND UP",
                            (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
                        )
        # 表示
        cv2.imshow("Pose Action (ESC to quit)", im)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
print("[INFO] finished.")
