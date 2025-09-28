# seatbelt_rule_demo.py
# ------------------------------------------------------------------------------------
# 使い方
#   uv run python seatbelt_rule_demo.py
#
# 必要環境
#   - Python 3.10–3.12
#   - uv で仮想環境 (推奨) / CPUでOK
#   - pip/uv で "ultralytics", "opencv-python" をインストール
#
# 入力の切替
#   VIDEO_SOURCE = "input.mp4"   # 動画ファイル
#   VIDEO_SOURCE = 0             # Webカメラ
#
# 画面の見方
#   - 腰付近に左右の ROI（四角）を表示。手首点（黄）が ROI に入ると Reach→Engage へ。
#   - 上部に現在の調整パラメータを表示（roi_scale / reach_secs / engage_secs / var_gain）。
#   - 各人物IDの上に フェーズ, ROI滞在フレーム数 を表示。
#   - Engage中は小刻み動作の "var" としきい値 "thr"、継続フレーム "dur" を表示。
#   - 検知成立時は「SEATBELT FASTENED」を**一定時間**（DISPLAY_SECS）表示し続けます。
#
# キー操作（ライブ調整）
#   [ / ]   : ROIスケール (肩幅×係数) を縮小/拡大    範囲: 0.10〜0.60
#   , / .   : reach_secs（Reach継続秒）を短く/長く   範囲: 0.05〜1.00
#   - / =   : engage_secs（Engage継続秒）を短く/長く 範囲: 0.20〜1.50
#   ; / '   : var_gain（小刻み度しきい値倍率）を下げ/上げ 範囲: 0.25〜3.00
#   s       : 検知成立フレームの保存 ON/OFF（./detected/ に保存）
#   q or ESC: 終了
#
# 出力
#   - 標準出力：検知ログ 例) [SEATBELT] frame=129 id=1 hand=R dur=54 var=1156.1 thr=32.0
#   - 画像保存：sキーでONにすると、検知フレームを ./detected/frame123_id1.jpg に保存
#
# アドバイス
#   検知しづらい場合は以下を順に調整：
#     1) roi_scale を 0.35→0.45 に上げる（手が ROI に入りやすく）
#     2) reach_secs を 0.25→0.15 に下げる（短い滞在で Reach）
#     3) engage_secs を 0.60→0.40 に下げる（早めに確定）
#     4) var_gain を 1.0→0.5 に下げる（小さな手の動きでも“ガチャ”扱い）
# ------------------------------------------------------------------------------------

from ultralytics import YOLO
import cv2, numpy as np, os
from collections import defaultdict, deque

# ====== 基本設定 ======
VIDEO_SOURCE = "input.mp4"     # 動画ファイル or 0（Webカメラ）
VIDEO_SOURCE = 0     # 動画ファイル or 0（Webカメラ）
MODEL_PATH   = "yolov8n-pose.pt"

# ライブ調整可能なパラメータ（起動時の初期値）
roi_scale   = 0.45   # 肩幅×係数 → ROI一辺のサイズ（0.35〜0.45くらいが目安）
reach_secs  = 0.15   # Reach判定に必要な連続滞在秒
engage_secs = 0.40   # Engage（“ガチャ”動作）の最短継続秒
var_gain    = 0.5   # “小刻み度”しきい値倍率（下げるほど検知しやすい）
PIX_MARGIN  = 12     # ROIに対する周辺マージン（px）

# 検知後のラベル保持秒数（一定時間表示）
DISPLAY_SECS = 1.5

# 検知フレームの保存（sキーで切替）
save_on_detect = False
detected_dir = "detected"

# ====== COCOキーポイントindex（YOLOv8 Pose準拠） ======
L_SH, R_SH = 5, 6
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12

# ====== 内部状態クラス ======
class PState:
    def __init__(self):
        self.phase = "Idle"
        self.in_roi_frames = 0
        self.wrist_history = deque(maxlen=90)
        self.engage_start = -1
        self.last_detect_frame = -99999  # 検知後の保持表示用

# ====== ユーティリティ ======
def draw_text(img, text, x, y, scale=0.6, color=(255,255,255)):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

def roi_from_hip(kps, W, H, scale_coef):
    """左右ヒップ中心に肩幅比例の正方形ROIを作る。"""
    l_sh, r_sh = kps[L_SH], kps[R_SH]
    l_hip, r_hip = kps[L_HIP], kps[R_HIP]
    if np.any(np.isnan([*l_sh,*r_sh,*l_hip,*r_hip])): 
        return None, None, 0.0
    shoulder_w = float(np.linalg.norm(r_sh - l_sh) + 1e-6)
    side = max(16.0, shoulder_w * scale_coef)
    def box(c):
        x1 = int(np.clip(c[0]-side/2, 0, W-1))
        y1 = int(np.clip(c[1]-side/2, 0, H-1))
        x2 = int(np.clip(c[0]+side/2, 0, W-1))
        y2 = int(np.clip(c[1]+side/2, 0, H-1))
        return [x1,y1,x2,y2]
    return box(l_hip), box(r_hip), shoulder_w

def in_box(pt, box):
    x,y = pt
    x1,y1,x2,y2 = box
    return (x1-PIX_MARGIN)<=x<= (x2+PIX_MARGIN) and (y1-PIX_MARGIN)<=y<= (y2+PIX_MARGIN)

def small_motion(history_xy):
    """ROI内の“ガチャ”っぽい小刻み動作を分散で近似。"""
    if len(history_xy) < 5: 
        return 0.0
    arr = np.array(history_xy, dtype=float)
    return float(np.var(arr, axis=0).sum())

# ====== メイン処理 ======
def main():
    global roi_scale, reach_secs, engage_secs, var_gain, save_on_detect

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError("動画/カメラを開けません。VIDEO_SOURCE を確認してください。")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    need_reach  = max(2, int(reach_secs  * fps))
    need_engage = max(3, int(engage_secs * fps))
    display_frames = max(1, int(DISPLAY_SECS * fps))

    print(f"[INFO] FPS={fps:.1f} reach_frames={need_reach} engage_frames={need_engage} display_frames={display_frames}")
    if save_on_detect and not os.path.exists(detected_dir):
        os.makedirs(detected_dir, exist_ok=True)

    states = defaultdict(PState)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]

        # YOLO 推論（追跡つき / デフォルトはByteTrack）
        for r in model.track(frame, stream=True, persist=True, verbose=False):
            im = r.plot()

            # 調整パラメータ（上部）表示
            draw_text(im, f"roi_scale[{roi_scale:.2f}]  reach[{reach_secs:.2f}s]  engage[{engage_secs:.2f}s]  var_gain[{var_gain:.2f}]",
                      10, 24, 0.55, (200,255,200))
            draw_text(im, f"[ / ] roi   ,/. reach   -/= engage   ;/' var   s:save={'ON' if save_on_detect else 'OFF'}   q/ESC: quit",
                      10, 48, 0.5, (180,220,255))

            if r.keypoints is None or r.boxes is None or r.boxes.id is None:
                # 追跡対象なしでもキー受付
                cv2.imshow("Seatbelt Demo", im)
                key = cv2.waitKey(1) & 0xFF
                if not handle_key(key, fps, states):
                    cap.release(); cv2.destroyAllWindows(); return
                # need_* などはキー入力後に更新される
                need_reach, need_engage = recompute_needs(fps)
                frame_idx += 1
                continue

            ids = r.boxes.id.cpu().numpy().astype(int)
            xy  = r.keypoints.xy.cpu().numpy()          # (N,17,2)
            boxes = r.boxes.xyxy.cpu().numpy()

            for i, tid in enumerate(ids):
                kps = xy[i]
                l_roi, r_roi, shoulder_w = roi_from_hip(kps, W, H, roi_scale)
                if l_roi is None:
                    continue

                # 近い手（左 or 右）を選ぶ
                lw, rw = kps[L_WR], kps[R_WR]
                dL = float(np.linalg.norm(lw - kps[L_HIP]))
                dR = float(np.linalg.norm(rw - kps[R_HIP]))
                hand = "R" if dR <= dL else "L"
                wrist = rw if hand=="R" else lw
                roi   = r_roi if hand=="R" else l_roi

                # 可視化（ROIと手首）
                cv2.rectangle(im, (roi[0],roi[1]), (roi[2],roi[3]), (0,170,200), 2)
                cv2.circle(im, tuple(np.int32(wrist)), 5, (0,255,255), -1)

                st = states[tid]
                inroi = in_box(wrist, roi)

                # フェーズ遷移
                if st.phase == "Idle":
                    st.in_roi_frames = st.in_roi_frames + 1 if inroi else 0
                    st.wrist_history.clear()
                    if st.in_roi_frames >= need_reach:
                        st.phase = "Reach"
                        st.engage_start = frame_idx

                elif st.phase == "Reach":
                    st.wrist_history.append(tuple(wrist))
                    if not inroi:
                        st.phase = "Idle"; st.in_roi_frames = 0; st.wrist_history.clear()
                    elif (frame_idx - st.engage_start) >= need_reach:
                        st.phase = "Engage"
                        st.engage_start = frame_idx  # Engage開始

                elif st.phase == "Engage":
                    st.wrist_history.append(tuple(wrist))
                    mot = small_motion(st.wrist_history)
                    # 肩幅スケールに基づく基準を倍率で調整
                    thr = var_gain * ((max(8.0, shoulder_w * 0.05)) ** 2)
                    dur = frame_idx - st.engage_start

                    # デバッグ情報（右上）
                    draw_text(im, f"ID {tid} ENG: dur={dur}/{need_engage}  var={mot:.1f}  thr={thr:.1f}",
                              10, 72, 0.55, (0,255,255))

                    if not inroi:
                        # ROIを離脱 → 条件を満たしていれば確定
                        if dur >= need_engage and mot >= thr:
                            st.last_detect_frame = frame_idx  # 以降DISPLAY_SECS表示
                            x1,y1,_,_ = boxes[i].astype(int)
                            draw_text(im, f"ID {tid}: SEATBELT FASTENED", x1, max(0,y1-10), 0.8, (0,255,0))
                            print(f"[SEATBELT] frame={frame_idx} id={tid} hand={hand} dur={dur} var={mot:.1f} thr={thr:.1f}")
                            # 検知フレーム保存（任意）
                            if save_on_detect:
                                os.makedirs(detected_dir, exist_ok=True)
                                out_path = os.path.join(detected_dir, f"frame{frame_idx}_id{tid}.jpg")
                                cv2.imwrite(out_path, im)
                        # どちらにせよリセット
                        st.phase = "Idle"; st.in_roi_frames = 0; st.wrist_history.clear()

                # IDとフェーズ表示（各ボックス上）
                x1,y1,_,_ = boxes[i].astype(int)
                draw_text(im, f"ID {tid} {st.phase} inROI={st.in_roi_frames}", x1, max(0, y1-30), 0.6, (255,200,0))

                # 検知後の“保持表示”
                if (frame_idx - st.last_detect_frame) <= display_frames:
                    draw_text(im, "SEATBELT FASTENED", 50, 50, 1.0, (0,255,0))

            # 表示 & キー処理
            cv2.imshow("Seatbelt Demo", im)
            key = cv2.waitKey(1) & 0xFF
            if not handle_key(key, fps, states):
                cap.release(); cv2.destroyAllWindows(); return
            need_reach, need_engage = recompute_needs(fps)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

def recompute_needs(fps):
    """秒→フレーム換算の更新（キー操作後に呼ぶ）"""
    need_reach  = max(2, int(reach_secs  * fps))
    need_engage = max(3, int(engage_secs * fps))
    return need_reach, need_engage

def handle_key(key, fps, states):
    """キー入力ハンドラ（True継続 / False終了）"""
    global roi_scale, reach_secs, engage_secs, var_gain, save_on_detect

    if key in (27, ord('q')):   # ESC / q
        return False
    elif key == ord('['):       # ROI縮小
        roi_scale = max(0.10, roi_scale - 0.02)
    elif key == ord(']'):       # ROI拡大
        roi_scale = min(0.60, roi_scale + 0.02)
    elif key == ord(','):       # Reach短く
        reach_secs = max(0.05, reach_secs - 0.02)
    elif key == ord('.'):       # Reach長く
        reach_secs = min(1.00, reach_secs + 0.02)
    elif key == ord('-'):       # Engage短く
        engage_secs = max(0.20, engage_secs - 0.05)
    elif key == ord('='):       # Engage長く
        engage_secs = min(1.50, engage_secs + 0.05)
    elif key == ord(';'):       # var_gain↓
        var_gain = max(0.25, var_gain - 0.1)
    elif key == ord('\''):      # var_gain↑
        var_gain = min(3.00, var_gain + 0.1)
    elif key == ord('s'):       # 保存ON/OFF
        save_on_detect = not save_on_detect
        if save_on_detect and not os.path.exists(detected_dir):
            os.makedirs(detected_dir, exist_ok=True)

    # 検知保持表示のために current frame を維持（states はそのまま）
    return True

if __name__ == "__main__":
    main()
