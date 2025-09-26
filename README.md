# Yoloの概要
YOLO（You Only Look Once）は、画像を一度だけ見て（1回のネットワーク推論で）物体の位置とクラスを同時に出す“1段（one-stage）型”の物体検出モデルです。R-CNN系のように候補領域→分類という段階を踏まないので、とにかく高速。初期の論文でも「リアルタイムで検出できる」ことが強調されています。

### よく使われるシーン
- 監視・防犯（人物/車両検出、侵入検知）
- 産業・製造（外観検査、不良検出、在庫カウント）
- 小売・物流（棚割、ピッキング支援）
- 自動運転/ADASの一部（歩行者・標識・車両検出）
- スポーツ・放送（トラッキングと組み合わせて分析）
- ロボティクス（把持対象の検出）YOLO系は検出だけでなく、セグメンテーションやポーズ推定、トラッキングなどの周辺タスクにも広く使われています。

### 特徴・強み
- 速い：1回の推論でボックスとクラスを出すので低レイテンシ。
- シンプル：エンドツーエンドで最適化しやすい。
- 実用の厚さ：豊富な事前学習モデルとツール群（学習・推論・エクスポート）。

### 最近の系統
- Ultralytics YOLO（v8/11 など）：使いやすいCLI/Python APIで、検出・分割・姿勢・追跡に対応。学習・推論・エクスポートが統一UIで楽。
- YOLOv10（THU）：NMS（非極大抑制）不要な学習戦略と設計で、精度-効率のバランスをさらに改善。Ultralyticsのパッケージからも利用できます。

# Yolo簡単なDemo

### 事前準備
1. Pythonをインストール
2. uvをインストール、インストール後 `uv --version` が通ればOK
    ```
    pip install uv
    ```
3. Projectのフォルダを作成
4. 仮想環境構築
   - CurrentフォルダをProjectフォルダに変更
   - 初期化
        ```
        uv init
        ```
   - 仮想環境を作成（.venv フォルダに）
        ```
        uv venv .venv
        ```
   - 依存を追加
        ```
        uv add ultralytics opencv-python
        ```

### 画像1枚で動作確認
`single_image.py`
```python
from ultralytics import YOLO
import cv2

# 軽量モデルをロード（初回は自動DL）
model = YOLO("yolov10n.pt")

# 好きな画像パス or URL（URLもOK）
img_path = "https://ultralytics.com/images/bus.jpg"

# 推論（結果は list[ultralytics.engine.results.Results]）
results = model(img_path)

# 1件目を描画して保存
res = results[0]
vis = res.plot()  # NumPy配列(BGR)
cv2.imwrite("out.jpg", vis)
print("saved -> out.jpg")
```

### Webカメラでリアルタイム推論（CPU）
`webcam_yolo.py`
```python
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
```

### Webカメラでリアルタイム推論（CPU）
`uv run yolo predict model=yolov10n.pt source="video.mp4" save=True`

### 公式の小さなCOCOサンプル（coco8.yaml）で学習の流れ
Ultralytics YOLO が用意している サンプルデータセット coco8.yaml を使っています。これは、本番用の巨大なCOCOデータセット（約118k枚）ではなく、学習の流れを確認するために作られた8クラス×各クラス数枚レベルの超小規模データセットです。
- coco8.yaml の中身
  - クラス（8種類）
    - person
    - bicycle
    - car
    - motorcycle
    - airplane
    - bus
    - train
    - truck
  - データ数
    - 学習用：約40枚
    - 検証用：約40枚
    - 合計：80枚程度
  - アノテーション形式
    - 物体ごとの矩形ボックス（bounding box）＋ラベル
- 学習の目的
本番モデルを作るためではなく、以下を体験するための「チュートリアルデータ」です：
   - 学習パイプラインの流れ
（前処理 → バッチ分割 → forward/backward → ロス計算 → 重み更新 → 評価）
   - ログや可視化の確認方法
（mAP、precision/recall、損失曲線など）
   - 重みファイルの生成と利用
（学習後の runs/detect/train*/weights/last.pt を推論に使える）
- Powershellコマンド
`uv run yolo train model=yolov10n.yaml data=coco8.yaml epochs=10 imgsz=640`
終わると `runs\detect\train*` に学習結果（重みや指標）が出ます。

### エクスポート（CPU配備向け）
#### Ultralytics YOLO の標準モデルは PyTorch形式 (.pt) です。ですが、実運用では以下のようなケースがあります：
- PyTorchが入っていない環境
  - 工場の検査機、組み込み機器、クラウド推論サーバなど
  - PyTorchはサイズが大きい＆起動が重い → 軽量な推論エンジンに変換したい
- 推論速度をもっと上げたい
  - NVIDIA GPUなら TensorRT に変換すると大幅に高速化できる
  - Intel CPU/Movidius/NPUなら OpenVINO が有利
  - モバイルなら CoreML（iOS）や TFLite（Android）
- 他のフレームワークやツールに統合したい
  - ONNX形式なら多くのAI推論フレームワークで共通に使える

#### 主なエクスポート形式と用途
|フォーマット|拡張子|主な用途|
|----|----|----|
|ONNX|.onnx|汎用。多くの推論エンジンで使える（onnxruntime, OpenVINO, TensorRTなど）|
|TensorRT|.engine|NVIDIA GPUで最速。組み込みGPU推論やクラウド推論に最適|
|OpenVINO|ディレクトリ形式|Intel CPU / VPU（Myriad, Movidius）向け最適化|
|CoreML|.mlmodel|iOS / macOS アプリ（Apple Neural Engine対応）|
|TensorFlow Lite (TFLite)|.tflite|Android / Edge TPU（Coral）|
|TorchScript|.pt|PyTorch依存だがC++/LibTorchで実行可能|
|Caffe / Paddle / ncnn|（形式ごとに異なる）|特殊な環境や中国圏のモバイル環境|

#### 具体例（コマンド）
```powershell
# ONNXに変換
uv run yolo export model=runs/detect/train/weights/best.pt format=onnx

# TensorRTに変換
uv run yolo export model=runs/detect/train/weights/best.pt format=engine

# OpenVINOに変換
uv run yolo export model=runs/detect/train/weights/best.pt format=openvino
```