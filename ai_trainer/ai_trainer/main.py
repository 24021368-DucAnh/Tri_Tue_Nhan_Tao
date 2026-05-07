# main.py — Integration & UI  (v4 — Workout Plan mode)
# Phím: Q=thoát | R=reset | ESC=về màn hình chọn plan

from __future__ import annotations
import cv2, pickle, numpy as np, os, time
from pose_extractor import PoseExtractor
from workout_plans  import WORKOUT_PLANS, WorkoutSession
from config         import WINDOW_W, WINDOW_H, SVM_PATH

# ── Màu sắc ──────────────────────────────────────────────────────────────────
C_CYAN   = (255, 220,  50)
C_WHITE  = (255, 255, 255)
C_GREEN  = ( 50, 255, 100)
C_RED    = ( 50,  50, 220)
C_GRAY   = (120, 120, 120)
C_DARK   = ( 15,  15,  15)
C_DONE   = ( 50, 255, 180)
C_GOLD   = ( 30, 215, 255)

# ── State ─────────────────────────────────────────────────────────────────────
STATE_SELECT  = "select"
STATE_WORKOUT = "workout"
STATE_DONE    = "done"


# ─────────────────────────────────────────────────────────────────────────────
# LOAD SVM
# ─────────────────────────────────────────────────────────────────────────────

def load_svm():
    if not os.path.exists(SVM_PATH):
        return None, None
    with open(SVM_PATH, "rb") as f:
        b = pickle.load(f)
    return b["model"], b["scaler"]

def predict(model, scaler, features):
    if model is None or features is None:
        return "idle", 0.0
    X     = scaler.transform([features])
    label = model.predict(X)[0]
    conf  = float(model.predict_proba(X)[0].max())
    return label, conf


# ─────────────────────────────────────────────────────────────────────────────
# HELPER VẼ
# ─────────────────────────────────────────────────────────────────────────────

def blend(frame, x1, y1, x2, y2, color, alpha=0.65):
    roi          = frame[y1:y2, x1:x2]
    overlay      = np.full_like(roi, 0)
    overlay[:]   = color
    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, overlay, alpha, 0)

def txt(frame, text, x, y, scale=0.7, color=C_WHITE, bold=False):
    cv2.putText(frame, str(text), (x, y),
                cv2.FONT_HERSHEY_DUPLEX, scale, color,
                2 if bold else 1, cv2.LINE_AA)

def fmt_time(secs: int) -> str:
    return f"{secs // 60:02d}:{secs % 60:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# MÀN HÌNH CHỌN PLAN
# ─────────────────────────────────────────────────────────────────────────────

def draw_select_screen(frame, w, h):
    blend(frame, 0, 0, w, h, C_DARK, alpha=0.82)

    txt(frame, "AI TRAINER", w // 2 - 140, 70,
        scale=1.8, color=C_CYAN, bold=True)
    txt(frame, "Chon workout plan  (nhan phim so)",
        w // 2 - 220, 115, scale=0.7, color=C_GRAY)

    card_w = min(500, w - 80)
    card_x = w // 2 - card_w // 2

    for i, (key, plan) in enumerate(WORKOUT_PLANS.items()):
        cy = 155 + i * 100
        # Nền card
        blend(frame, card_x, cy, card_x + card_w, cy + 80, (30, 30, 30), alpha=0.85)
        cv2.rectangle(frame, (card_x, cy), (card_x + card_w, cy + 80), C_GRAY, 1)

        # Số phím
        blend(frame, card_x, cy, card_x + 58, cy + 80, (40, 40, 40), alpha=0.9)
        txt(frame, key, card_x + 18, cy + 50, scale=1.2, color=C_CYAN, bold=True)

        # Tên plan
        txt(frame, plan["name"], card_x + 70, cy + 30,
            scale=0.85, color=C_WHITE, bold=True)

        # Danh sách bài
        ex_names = [s["exercise"].replace("_", " ") for s in plan["exercises"]]
        txt(frame, "  •  ".join(ex_names),
            card_x + 70, cy + 60, scale=0.5, color=C_GRAY)

        # Tổng số bài
        total_ex = len(plan["exercises"])
        txt(frame, f"{total_ex} bai",
            card_x + card_w - 80, cy + 50, scale=0.6, color=C_CYAN)

    txt(frame, "Q = thoat",
        w // 2 - 45, h - 25, scale=0.55, color=C_GRAY)


# ─────────────────────────────────────────────────────────────────────────────
# HUD WORKOUT
# ─────────────────────────────────────────────────────────────────────────────

def draw_skeleton(frame, lm, w, h):
    CONNS = [(11,13),(13,15),(12,14),(14,16),(11,12),
             (11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]
    for a, b in CONNS:
        p1 = (int(lm[a].x * w), int(lm[a].y * h))
        p2 = (int(lm[b].x * w), int(lm[b].y * h))
        cv2.line(frame, p1, p2, (200, 200, 200), 2)
    for i in [11,12,13,14,15,16,23,24,25,26,27,28]:
        cx, cy = int(lm[i].x * w), int(lm[i].y * h)
        cv2.circle(frame, (cx, cy), 5, C_CYAN,  -1)
        cv2.circle(frame, (cx, cy), 7, C_WHITE,  1)


def draw_workout_hud(frame, d: dict, svm_label: str, conf: float, w, h):
    """d = dict trả về từ WorkoutSession.update()"""
    exercise = d.get("exercise", "idle")
    rep      = d.get("rep", 0)
    tr       = d.get("target_reps", 1)
    cur_set  = d.get("set", 1)
    ts       = d.get("target_sets", 1)
    pct      = d.get("percent_complete", 0.0)
    angle    = d.get("angle", 0.0)
    stage    = d.get("stage") or "-"
    si       = d.get("step_idx", 0)
    total_st = d.get("total_steps", 1)
    steps    = d.get("steps", [])
    done_arr = d.get("completed_steps", [])
    elapsed  = d.get("elapsed_seconds", 0)

    # ── Thanh trên ───────────────────────────────────────────────────────
    blend(frame, 0, 0, w, 55, C_DARK, alpha=0.78)
    txt(frame, "AI TRAINER", 14, 38, scale=0.9, color=C_CYAN, bold=True)
    txt(frame, d.get("plan_name", ""), 180, 35, scale=0.65, color=C_GRAY)
    txt(frame, fmt_time(elapsed), w - 110, 38, scale=0.8, color=C_WHITE, bold=True)
    txt(frame, f"Bai {si + 1}/{total_st}", w - 210, 38, scale=0.6, color=C_GRAY)
    cv2.line(frame, (0, 55), (w, 55), C_CYAN, 1)

    # ── Panel trái: bài đang tập ─────────────────────────────────────────
    blend(frame, 0, 65, 270, 330, C_DARK, alpha=0.7)
    txt(frame, "EXERCISE",              14,  92, scale=0.5, color=C_GRAY)
    txt(frame, exercise.upper(),        14, 128, scale=0.95, color=C_CYAN, bold=True)
    txt(frame, f"SVM: {svm_label} {conf*100:.0f}%",
        14, 155, scale=0.5, color=C_GRAY)

    txt(frame, "ANGLE",                 14, 190, scale=0.5, color=C_GRAY)
    txt(frame, f"{angle:.0f} deg",      14, 225, scale=0.95, color=C_WHITE, bold=True)
    txt(frame, f"Stage: {stage.upper()}",14,258,scale=0.65,
        color=C_GREEN if stage == "up" else C_CYAN)

    # ── Panel phải: rep + set ────────────────────────────────────────────
    blend(frame, w - 240, 65, w, 240, C_DARK, alpha=0.7)
    txt(frame, "REPS",          w-228,  92, scale=0.5,  color=C_GRAY)
    txt(frame, f"{rep} / {tr}", w-228, 138, scale=1.2,  color=C_WHITE, bold=True)
    txt(frame, "SET",           w-228, 168, scale=0.5,  color=C_GRAY)
    txt(frame, f"{cur_set} / {ts}", w-228, 210, scale=1.2, color=C_WHITE, bold=True)

    # ── Sidebar phải: danh sách bài tập ─────────────────────────────────
    sb_x = w - 240
    blend(frame, sb_x, 248, w, 248 + len(steps) * 52 + 20, C_DARK, alpha=0.7)
    txt(frame, "WORKOUT PLAN", sb_x + 10, 270, scale=0.5, color=C_GRAY)

    for i, step in enumerate(steps):
        row_y   = 292 + i * 52
        is_cur  = (i == si)
        is_done = done_arr[i] if i < len(done_arr) else False

        # Highlight bài đang tập
        if is_cur:
            blend(frame, sb_x + 6, row_y - 18, w - 6, row_y + 28,
                  (40, 80, 40), alpha=0.7)
            cv2.rectangle(frame,
                          (sb_x + 6, row_y - 18), (w - 6, row_y + 28),
                          C_GREEN, 1)

        icon  = "✓" if is_done else ("▶" if is_cur else "○")
        color = C_DONE if is_done else (C_GREEN if is_cur else C_GRAY)
        name  = step["exercise"].replace("_", " ").title()
        txt(frame, f"{icon} {name}", sb_x + 14, row_y + 6,
            scale=0.55, color=color, bold=is_cur)
        txt(frame, f"{step['reps']}r x {step['sets']}s",
            sb_x + 14, row_y + 24, scale=0.45, color=C_GRAY)

    # ── Progress bar ─────────────────────────────────────────────────────
    bx1, bx2, by, bh = 14, w - 254, h - 28, 16
    blend(frame, bx1, by - bh, bx2, by + 4, C_DARK, alpha=0.85)
    cv2.rectangle(frame, (bx1, by - bh), (bx2, by + 4), C_GRAY, 1)
    fill = int(bx1 + (bx2 - bx1) * pct / 100)
    bc   = C_DONE if pct >= 100 else C_CYAN
    if fill > bx1:
        cv2.rectangle(frame, (bx1+1, by-bh+1), (fill, by+3), bc, -1)
    txt(frame, f"{pct:.0f}%", bx1 + 8, by - 1, scale=0.48, color=C_WHITE)
    cv2.line(frame, (0, h - 50), (w - 248, h - 50), C_GRAY, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MÀN HÌNH CHÚC MỪNG
# ─────────────────────────────────────────────────────────────────────────────

def draw_congrats(frame, d: dict, w, h):
    blend(frame, 0, 0, w, h, (10, 30, 10), alpha=0.88)

    # Viền nháy vàng
    t      = time.time()
    border = C_GOLD if int(t * 2) % 2 == 0 else C_DONE
    cv2.rectangle(frame, (8, 8), (w - 8, h - 8), border, 4)

    cy = h // 2 - 120
    txt(frame, "WORKOUT COMPLETE!",
        w // 2 - 270, cy, scale=1.8, color=C_GOLD, bold=True)
    txt(frame, d.get("plan_name", ""),
        w // 2 - 150, cy + 55, scale=0.9, color=C_DONE)

    # Stats
    elapsed = d.get("elapsed_seconds", 0)
    total_r = d.get("total_reps", 0)
    steps   = d.get("total_steps", 0)

    for i, (label, val) in enumerate([
        ("Thoi gian",   fmt_time(elapsed)),
        ("Tong reps",   str(total_r)),
        ("So bai tap",  str(steps)),
    ]):
        bx = w // 2 - 300 + i * 210
        by = cy + 110
        blend(frame, bx, by, bx + 190, by + 90, (20, 50, 20), alpha=0.8)
        cv2.rectangle(frame, (bx, by), (bx + 190, by + 90), C_DONE, 1)
        txt(frame, label,  bx + 12, by + 28, scale=0.55, color=C_GRAY)
        txt(frame, val,    bx + 12, by + 68, scale=1.1,  color=C_DONE, bold=True)

    txt(frame, "Nhan ESC de chon plan moi  |  Q de thoat",
        w // 2 - 240, cy + 240, scale=0.65, color=C_GRAY)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    model, scaler = load_svm()
    extractor     = PoseExtractor(mode="VIDEO")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WINDOW_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)
    cv2.namedWindow("AI Trainer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Trainer", WINDOW_W, WINDOW_H)

    state   : str                   = STATE_SELECT
    session : WorkoutSession | None = None
    svm_raw   = "idle"
    conf      = 0.0
    last_d    = {}

    print("Chay! Q=thoat | ESC=ve chon plan | R=reset")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w         = frame.shape[:2]
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Lấy pose
        features, lm_list = extractor.get_features(frame, timestamp_ms)
        svm_raw, conf     = predict(model, scaler, features)
        if lm_list:
            draw_skeleton(frame, lm_list, w, h)

        # ── STATE MACHINE ─────────────────────────────────────────────────
        if state == STATE_SELECT:
            draw_select_screen(frame, w, h)

        elif state == STATE_WORKOUT:
            lm = lm_list if lm_list else None
            last_d = session.update(lm)

            if last_d["is_complete"]:
                state = STATE_DONE
            else:
                draw_workout_hud(frame, last_d, svm_raw, conf, w, h)

        elif state == STATE_DONE:
            draw_congrats(frame, last_d, w, h)

        cv2.imshow("AI Trainer", frame)

        # ── PHÍM BẤM ─────────────────────────────────────────────────────
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            break

        elif key == 27:  # ESC → về màn hình chọn plan
            state   = STATE_SELECT
            session = None

        elif state == STATE_SELECT:
            k = chr(key) if 32 <= key < 128 else ""
            if k in WORKOUT_PLANS:
                session = WorkoutSession(WORKOUT_PLANS[k])
                state   = STATE_WORKOUT
                print(f"[INFO] Bat dau: {WORKOUT_PLANS[k]['name']}")

        elif state == STATE_WORKOUT and key == ord("r"):
            session = WorkoutSession(session.plan)   # restart cùng plan
            print("[INFO] Reset plan.")

    extractor.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
