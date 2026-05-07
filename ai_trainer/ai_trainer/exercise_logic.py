# exercise_logic.py — THÀNH VIÊN 3: Logic & Math
# Thêm: invert mode cho Shoulder Press, Lateral Raise, Jumping Jack

from __future__ import annotations

import math
from collections import deque, Counter
from config import EXERCISE_CONFIG, DEFAULT_TARGET_REPS, DEFAULT_TARGET_SETS, STABILITY_FRAMES


# ─────────────────────────────────────────────────────────────────────────────
# HÀM TÍNH GÓC 2D
# ─────────────────────────────────────────────────────────────────────────────

def calculate_angle(a, b, c) -> float:
    """Góc tại B (độ, 0–180). a/b/c là MediaPipe landmark có .x .y"""
    ax, ay = a.x - b.x, a.y - b.y
    cx, cy = c.x - b.x, c.y - b.y
    angle  = math.degrees(math.atan2(cy, cx) - math.atan2(ay, ax))
    angle  = abs(angle)
    return 360 - angle if angle > 180 else angle


# ─────────────────────────────────────────────────────────────────────────────
# REP COUNTER
# ─────────────────────────────────────────────────────────────────────────────

class RepCounter:
    """
    State machine đếm rep, hỗ trợ 2 chiều chuyển động:

    invert=False  (mặc định — góc GIẢM để tính rep):
        angle > down_angle  →  stage "down"   (vị trí thấp / duỗi)
        angle < up_angle    →  stage "up"     (vị trí cao / gập)  → count+1
        Bài: Bicep Curl, Squat, Push-up, High Knees

    invert=True  (góc TĂNG để tính rep):
        angle < down_angle  →  stage "down"   (vị trí bắt đầu / gập)
        angle > up_angle    →  stage "up"     (vị trí đỉnh / duỗi) → count+1
        Bài: Shoulder Press, Lateral Raise, Jumping Jack

    mode='any'  : bất kỳ bên nào hoàn thành → +1 (curl, high knees)
    mode='both' : tất cả bên cùng hoàn thành → +1 (squat, press...)
    """

    def __init__(self, exercise_name: str):
        cfg              = EXERCISE_CONFIG.get(exercise_name, {})
        self.joints      = cfg.get("joints",      [(11, 13, 15)])
        self.up_angle    = cfg.get("up_angle",    45)
        self.down_angle  = cfg.get("down_angle",  160)
        self.invert      = cfg.get("invert",      False)
        self.mode        = cfg.get("mode",        "any")
        self.count       = 0
        self.stages      = [None]  * len(self.joints)  # None|"down"|"up"
        self._went_up    = [False] * len(self.joints)  # dùng cho mode=both

    def update(self, landmarks) -> tuple[int, str, float]:
        """Trả về (count, stage_display, avg_angle)."""
        angles   = []
        new_reps = []

        for i, (ia, ib, ic) in enumerate(self.joints):
            angle = calculate_angle(landmarks[ia], landmarks[ib], landmarks[ic])
            angles.append(angle)

            if not self.invert:
                # Góc GIẢM: down = góc lớn, up = góc nhỏ
                if angle > self.down_angle:
                    self.stages[i]   = "down"
                    self._went_up[i] = False
                if angle < self.up_angle and self.stages[i] == "down":
                    self.stages[i]   = "up"
                    self._went_up[i] = True
                    new_reps.append(i)
            else:
                # Góc TĂNG: down = góc nhỏ, up = góc lớn
                if angle < self.down_angle:
                    self.stages[i]   = "down"
                    self._went_up[i] = False
                if angle > self.up_angle and self.stages[i] == "down":
                    self.stages[i]   = "up"
                    self._went_up[i] = True
                    new_reps.append(i)

        # Đếm theo mode
        if self.mode == "any":
            self.count += len(new_reps)
        elif self.mode == "both" and all(self._went_up):
            self.count      += 1
            self._went_up    = [False] * len(self.joints)

        avg_angle = sum(angles) / len(angles)
        stage     = self.stages[0] or "idle"
        return self.count, stage, avg_angle

    def reset(self):
        self.count    = 0
        self.stages   = [None]  * len(self.joints)
        self._went_up = [False] * len(self.joints)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class SessionManager:
    """
    - Stability buffer: chỉ đổi bài sau STABILITY_FRAMES frame đồng thuận
    - Rep memory: lưu count mỗi bài, không mất khi tạm lẫn nhãn
    """

    def __init__(self,
                 target_reps: int = DEFAULT_TARGET_REPS,
                 target_sets: int = DEFAULT_TARGET_SETS):
        self.target_reps   = target_reps
        self.target_sets   = target_sets
        self.current_set   = 1
        self._confirmed_ex = "idle"
        self._pred_buffer  = deque(maxlen=STABILITY_FRAMES)
        self._counter      = None
        self._rep_memory   = {}   # {exercise_name: count}

    # ── PUBLIC ──────────────────────────────────────────────────────────

    def push_prediction(self, exercise_name: str) -> str:
        """
        Nhận dự đoán SVM mỗi frame.
        Trả về tên bài đang CHÍNH THỨC chạy (sau khi lọc stability).
        """
        self._pred_buffer.append(exercise_name)

        if len(self._pred_buffer) < STABILITY_FRAMES:
            return self._confirmed_ex

        most_common, freq = Counter(self._pred_buffer).most_common(1)[0]
        if most_common != self._confirmed_ex and freq >= int(STABILITY_FRAMES * 0.7):
            self._switch_exercise(most_common)

        return self._confirmed_ex

    def update(self, landmarks) -> dict:
        """Cập nhật logic đếm rep, trả về dict cho UI."""
        result = {
            "exercise":         self._confirmed_ex,
            "rep":              0,
            "set":              self.current_set,
            "target_reps":      self.target_reps,
            "target_sets":      self.target_sets,
            "percent_complete": 0.0,
            "set_done":         False,
            "angle":            0.0,
            "stage":            None,
        }

        if self._counter is None or landmarks is None:
            return result

        rep, stage, angle = self._counter.update(landmarks)
        result.update({"rep": rep, "stage": stage, "angle": angle})
        result["percent_complete"] = min(rep / self.target_reps * 100, 100.0)

        if rep >= self.target_reps:
            result["set_done"] = True
            if self.current_set < self.target_sets:
                self.current_set += 1
            self._rep_memory[self._confirmed_ex] = 0
            self._counter.reset()

        return result

    def reset_session(self):
        self.current_set = 1
        self._rep_memory = {}
        self._pred_buffer.clear()
        if self._counter:
            self._counter.reset()

    # ── PRIVATE ─────────────────────────────────────────────────────────

    def _switch_exercise(self, new_ex: str):
        if self._counter is not None:
            self._rep_memory[self._confirmed_ex] = self._counter.count

        self._confirmed_ex = new_ex
        if new_ex == "idle":
            self._counter = None
            return

        self._counter       = RepCounter(new_ex)
        self._counter.count = self._rep_memory.get(new_ex, 0)
