from ultralytics import YOLO
import cv2

# 軽量モデルをロード（初回は自動DL）
model = YOLO("yolov10n.pt")

# 好きな画像パス or URL（URLもOK）
img_path = "https://shinkyokushin-shoguchi.com/images/guide/1.jpg"

# 推論（結果は list[ultralytics.engine.results.Results]）
results = model(img_path)

# 1件目を描画して保存
res = results[0]
vis = res.plot()  # NumPy配列(BGR)
cv2.imwrite("out.jpg", vis)
print("saved -> out.jpg")
