from ultralytics import YOLO
import cv2

# 軽量モデル（n=Nano）。CPUでも軽快
model = YOLO("yolov10n.pt")

# カメラを開く（内蔵/USB。番号は0,1...で切替）
cap = cv2.VideoCapture(0)  # 必要なら cv2.CAP_DSHOW を指定: cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Webカメラを開けません。番号やドライバを確認してください。")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # ストリーム推論（メモリ効率◎）
    for r in model(frame, stream=True):
        vis = r.plot()  # 検出結果を描画（BGR配列）
        cv2.imshow("YOLOv10 (ESCで終了)", vis)

    # ESCで終了
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
