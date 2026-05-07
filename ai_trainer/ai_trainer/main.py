# main.py — THÀNH VIÊN 4: Integration & UI

import cv2
import pickle
import numpy as np
import os
from pose_extractor import PoseExtractor
from exercise_logic import SessionManager
from config import WINDOW_W, WINDOW_H, SVM_PATH

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL SVM
# ─────────────────────────────────────────────────────────────────────────────

def load_svm():
    if not os.path.exists(SVM_PATH):
        print("[WARN] Chưa có svm_model.pkl → chạy train_svm.py trước.")
        print("       Tạm thời dùng label mặc định: idle")
        return None, None
    with open(SVM_PATH, "rb") as f:
        bundle = pickle.load(f)
    print("[INFO] Đã load SVM model.")
    return bundle["model"], bundle["scaler"]


def predict_exercise(model, scaler, features):
    """Trả về (label, confidence). Nếu chưa có model → 'idle'."""
    if model is None or features is None:
        return "idle", 0.0
    X    = scaler.transform([features])
    label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return label, float(proba.max())


# ─────────────────────────────────────────────────────────────────────────────
# HUD — FUTURISTIC UI
# ─────────────────────────────────────────────────────────────────────────────

C_CYAN  = (255, 220,  50)
C_WHITE = (255, 255, 255)
C_GREEN = ( 50, 255, 100)
C_GRAY  = (120, 120, 120)
C_BG    = ( 15,  15,  15)
C_DONE  = ( 50, 255, 180)


def _blend(frame, x1, y1, x2, y2, color, alpha=0.6):
    roi     = frame[y1:y2, x1:x2]
    solid   = np.full_like(roi, 0)
    solid[:] = color
    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, solid, alpha, 0)


def _txt(frame, text, x, y, scale=0.7, color=C_WHITE, bold=False):
    thick = 2 if bold else 1
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)


def draw_hud(frame, exercise, confidence, svm_raw_label, session_data, w, h):
    # ── Thanh trên ──────────────────────────────────────────────────────
    _blend(frame, 0, 0, w, 58, C_BG, alpha=0.75)
    _txt(frame, "AI TRAINER",  12,  38, scale=1.0, color=C_CYAN, bold=True)
    _txt(frame, f"SVM raw: {svm_raw_label}  |  Confirmed: {exercise}",
         240, 35, scale=0.55, color=C_GRAY)
    cv2.line(frame, (0, 58), (w, 58), C_CYAN, 1)

    # ── Panel trái: bài tập ──────────────────────────────────────────────
    _blend(frame, 0, 68, 265, 310, C_BG, alpha=0.65)
    _txt(frame, "EXERCISE",              12,  95, scale=0.5,  color=C_GRAY)
    _txt(frame, exercise.upper(),        12, 130, scale=1.0,  color=C_CYAN, bold=True)
    conf_c = C_GREEN if confidence > 0.75 else C_CYAN
    _txt(frame, f"Conf: {confidence*100:.0f}%",
         12, 158, scale=0.6, color=conf_c)
    _txt(frame, "ANGLE",                 12, 192, scale=0.5,  color=C_GRAY)
    _txt(frame, f"{session_data.get('angle', 0):.0f} deg",
         12, 228, scale=1.0, color=C_WHITE, bold=True)
    stage = session_data.get("stage") or "-"
    _txt(frame, f"Stage: {stage.upper()}",
         12, 258, scale=0.65,
         color=C_GREEN if stage == "up" else C_CYAN)

    # ── Panel phải: rep / set ────────────────────────────────────────────
    _blend(frame, w - 245, 68, w, 250, C_BG, alpha=0.65)
    rep = session_data.get("rep", 0)
    tr  = session_data.get("target_reps", 12)
    _txt(frame, "REPS",          w - 233,  95, scale=0.5,  color=C_GRAY)
    _txt(frame, f"{rep} / {tr}", w - 233, 138, scale=1.2, color=C_WHITE, bold=True)
    cur = session_data.get("set", 1)
    ts  = session_data.get("target_sets", 3)
    _txt(frame, "SET",           w - 233, 172, scale=0.5,  color=C_GRAY)
    _txt(frame, f"{cur} / {ts}", w - 233, 210, scale=1.2, color=C_WHITE, bold=True)

    # ── Progress bar ─────────────────────────────────────────────────────
    pct   = session_data.get("percent_complete", 0.0)
    bx1, bx2, by = 20, w - 20, h - 28
    bh    = 16
    _blend(frame, bx1, by - bh, bx2, by + 4, C_BG, alpha=0.8)
    cv2.rectangle(frame, (bx1, by - bh), (bx2, by + 4), C_GRAY, 1)
    fill = int(bx1 + (bx2 - bx1) * pct / 100)
    bar_c = C_DONE if pct >= 100 else C_CYAN
    if fill > bx1:
        cv2.rectangle(frame, (bx1 + 1, by - bh + 1), (fill, by + 3), bar_c, -1)
    _txt(frame, f"{pct:.0f}%", bx1 + 8, by - 2, scale=0.5, color=C_WHITE)
    cv2.line(frame, (0, h - 50), (w, h - 50), C_GRAY, 1)

    # ── SET COMPLETE banner ───────────────────────────────────────────────
    if session_data.get("set_done"):
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), C_DONE, 4)
        _txt(frame, "SET COMPLETE!", w // 2 - 130, h // 2,
             scale=2.0, color=C_DONE, bold=True)


def draw_skeleton(frame, lm_list, w, h):
    CONNS = [(11,13),(13,15),(12,14),(14,16),(11,12),
             (11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]
    for a, b in CONNS:
        p1 = (int(lm_list[a].x * w), int(lm_list[a].y * h))
        p2 = (int(lm_list[b].x * w), int(lm_list[b].y * h))
        cv2.line(frame, p1, p2, (200, 200, 200), 2)
    for i in [11,12,13,14,15,16,23,24,25,26,27,28]:
        cx = int(lm_list[i].x * w)
        cy = int(lm_list[i].y * h)
        cv2.circle(frame, (cx, cy), 5, C_CYAN,  -1)
        cv2.circle(frame, (cx, cy), 7, C_WHITE,  1)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    model, scaler = load_svm()
    extractor     = PoseExtractor(mode="VIDEO")
    session       = SessionManager()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WINDOW_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)

    cv2.namedWindow("AI Trainer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Trainer", WINDOW_W, WINDOW_H)

    svm_raw    = "idle"
    confidence = 0.0
    print("Chạy! Q = thoát | R = reset session")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w         = frame.shape[:2]
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Bước 1: Trích features
        features, lm_list = extractor.get_features(frame, timestamp_ms)

        # Bước 2: SVM dự đoán (raw — có thể lẫn lộn)
        svm_raw, confidence = predict_exercise(model, scaler, features)

        # Bước 3: Đưa vào stability buffer → lấy nhãn ổn định
        confirmed_ex = session.push_prediction(svm_raw)

        # Bước 4: Cập nhật logic đếm rep
        session_data = session.update(lm_list)

        # Bước 5: Vẽ
        if lm_list is not None:
            draw_skeleton(frame, lm_list, w, h)

        draw_hud(frame, confirmed_ex, confidence, svm_raw, session_data, w, h)

        cv2.imshow("AI Trainer", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            session.reset_session()
            print("[INFO] Reset session.")

    extractor.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
