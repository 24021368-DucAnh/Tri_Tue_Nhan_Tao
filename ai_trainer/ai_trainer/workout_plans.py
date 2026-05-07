# workout_plans.py — Định nghĩa combo bài tập + WorkoutSession

from __future__ import annotations
import time
from exercise_logic import RepCounter

# ─────────────────────────────────────────────────────────────────────────────
# DANH SÁCH COMBO — thêm/sửa thoải mái ở đây
# ─────────────────────────────────────────────────────────────────────────────
# Mỗi plan gồm list "steps", mỗi step: {exercise, reps, sets}
# exercise phải khớp đúng tên trong EXERCISE_CONFIG của config.py

WORKOUT_PLANS = {
    "1": {
        "name":      "Beginner Full Body",
        "emoji":     "🟢",
        "exercises": [
            {"exercise": "squat",          "reps": 10, "sets": 2},
            {"exercise": "pushup",         "reps":  8, "sets": 2},
            {"exercise": "bicep_curl",     "reps": 12, "sets": 2},
        ],
    },
    "2": {
        "name":      "Cardio Blast",
        "emoji":     "🔴",
        "exercises": [
            {"exercise": "jumping_jack",   "reps": 20, "sets": 2},
            {"exercise": "high_knees",     "reps": 20, "sets": 2},
            {"exercise": "squat",          "reps": 15, "sets": 2},
        ],
    },
    "3": {
        "name":      "Upper Body",
        "emoji":     "🔵",
        "exercises": [
            {"exercise": "shoulder_press", "reps": 10, "sets": 3},
            {"exercise": "lateral_raise",  "reps": 10, "sets": 3},
            {"exercise": "bicep_curl",     "reps": 12, "sets": 3},
            {"exercise": "pushup",         "reps": 10, "sets": 3},
        ],
    },
    "4": {
        "name":      "Lower Body",
        "emoji":     "🟡",
        "exercises": [
            {"exercise": "squat",          "reps": 15, "sets": 3},
            {"exercise": "high_knees",     "reps": 30, "sets": 3},
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# WORKOUT SESSION — quản lý tiến trình theo plan
# ─────────────────────────────────────────────────────────────────────────────

class WorkoutSession:
    """
    Chạy tuần tự các bài tập trong plan.

    Luồng:
        rep đủ target → sang set tiếp
        set đủ target → sang exercise tiếp (đánh dấu ✓)
        hết exercise  → is_complete = True → màn hình chúc mừng
    """

    def __init__(self, plan: dict):
        self.plan           = plan
        self.steps          = plan["exercises"]
        self.step_idx       = 0
        self.current_set    = 1
        self.is_complete    = False
        self.start_time     = time.time()
        self.total_reps     = 0
        self.completed_steps: list[bool] = [False] * len(self.steps)
        self._counter       = RepCounter(self.steps[0]["exercise"])

    # ── Property tiện ích ────────────────────────────────────────────────────

    @property
    def current_step(self) -> dict:
        return self.steps[self.step_idx]

    @property
    def current_exercise(self) -> str:
        return self.current_step["exercise"]

    @property
    def elapsed_seconds(self) -> int:
        return int(time.time() - self.start_time)

    # ── Update mỗi frame ─────────────────────────────────────────────────────

    def update(self, landmarks) -> dict:
        """Trả về dict trạng thái đầy đủ cho UI mỗi frame."""
        if self.is_complete or landmarks is None:
            return self._result(0, None, 0.0)

        rep, stage, angle = self._counter.update(landmarks)

        # Hoàn thành đủ rep của set này
        if rep >= self.current_step["reps"]:
            self.total_reps += rep

            if self.current_set < self.current_step["sets"]:
                # Còn set → sang set tiếp
                self.current_set += 1
                self._counter.reset()
            else:
                # Hết set → đánh dấu bài này xong, sang bài tiếp
                self.completed_steps[self.step_idx] = True
                self.step_idx  += 1
                self.current_set = 1

                if self.step_idx >= len(self.steps):
                    self.is_complete = True
                else:
                    self._counter = RepCounter(self.current_exercise)

            rep = 0  # reset hiển thị cho bước mới

        return self._result(rep, stage, angle)

    # ── Builder ──────────────────────────────────────────────────────────────

    def _result(self, rep: int, stage, angle: float) -> dict:
        """Dict trả về cho UI."""
        if self.is_complete:
            return {
                "is_complete":      True,
                "plan_name":        self.plan["name"],
                "elapsed_seconds":  self.elapsed_seconds,
                "total_reps":       self.total_reps,
                "total_steps":      len(self.steps),
            }

        step = self.current_step
        pct  = min(rep / step["reps"] * 100, 100.0)

        return {
            "is_complete":      False,
            "plan_name":        self.plan["name"],
            "exercise":         self.current_exercise,
            "rep":              rep,
            "target_reps":      step["reps"],
            "set":              self.current_set,
            "target_sets":      step["sets"],
            "step_idx":         self.step_idx,
            "total_steps":      len(self.steps),
            "percent_complete": pct,
            "stage":            stage,
            "angle":            angle,
            "completed_steps":  self.completed_steps,
            "steps":            self.steps,
            "elapsed_seconds":  self.elapsed_seconds,
        }
