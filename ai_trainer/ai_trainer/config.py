import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_full.task")
SVM_PATH   = os.path.join(BASE_DIR, "models", "svm_model.pkl")
DATA_CSV   = os.path.join(BASE_DIR, "data", "dataset.csv")
VIDEO_DIR  = os.path.join(BASE_DIR, "data", "videos")

SMOOTH_WINDOW = 5
WINDOW_W      = 1280
WINDOW_H      = 720

LABELS = [
    "bicep_curl",
    "squat",
    "pushup",
    "shoulder_press",
    "lateral_raise",
    "high_knees",
    "jumping_jack",
    "idle",
]

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
#
# joints      : list cặp (A, B, C) — tính góc tại B; mỗi cặp = 1 bên cơ thể
# up_angle    : ngưỡng góc "đỉnh chuyển động"
# down_angle  : ngưỡng góc "đáy chuyển động"
# invert      : False (đếm khi góc GIẢM)  |  True (đếm khi góc TĂNG)
# mode        : 'any'  = 1 bên hoàn thành → +1 rep  (curl 1 tay, high knees)
#               'both' = cả 2 bên → +1 rep           (squat, press, jack)
#
# Chiều đếm mặc định  (invert=False):
#   angle > down_angle  →  stage "down"
#   angle < up_angle    →  stage "up"  →  count + 1
#   Dùng cho: Bicep Curl, Squat, Push-up, High Knees
#
# Chiều đếm đảo ngược  (invert=True):
#   angle < down_angle  →  stage "down"
#   angle > up_angle    →  stage "up"  →  count + 1
#   Dùng cho: Shoulder Press, Lateral Raise, Jumping Jack
#   (các bài tập bắt đầu từ tư thế gập → đẩy/nâng lên)

EXERCISE_CONFIG = {

    # ── Bài cũ ──────────────────────────────────────────────────────────────

    "bicep_curl": {
        "joints": [
            (11, 13, 15),   # vai trái  → khuỷu trái  → cổ tay trái
            (12, 14, 16),   # vai phải  → khuỷu phải  → cổ tay phải
        ],
        "up_angle":   45,
        "down_angle": 160,
        "invert":     False,
        "mode":       "any",   # curl từng tay xen kẽ
    },

    "squat": {
        "joints": [
            (23, 25, 27),   # hông trái → gối trái → cổ chân trái
            (24, 26, 28),   # hông phải → gối phải → cổ chân phải
        ],
        "up_angle":   100,
        "down_angle": 165,
        "invert":     False,
        "mode":       "both",
    },

    "pushup": {
        "joints": [
            (11, 13, 15),
            (12, 14, 16),
        ],
        "up_angle":   90,
        "down_angle": 155,
        "invert":     False,
        "mode":       "both",
    },

    # ── Bài mới ─────────────────────────────────────────────────────────────

    "shoulder_press": {
        # Khớp: vai → khuỷu → cổ tay (giống curl nhưng chiều ngược lại)
        # Tư thế bắt đầu: tay gập ~90° (tạ ngang vai)
        # Tư thế đỉnh:    tay duỗi ~165° (tạ lên đầu)
        # → Đếm khi góc TĂNG (invert=True)
        "joints": [
            (11, 13, 15),
            (12, 14, 16),
        ],
        "down_angle": 90,    # tay gập = đáy chuyển động
        "up_angle":   160,   # tay duỗi = đỉnh chuyển động
        "invert":     True,
        "mode":       "both",
    },

    "lateral_raise": {
        # Khớp: hông → vai → khuỷu  →  đo góc dang cánh tay
        # Tư thế bắt đầu: tay thả dọc thân ~20°
        # Tư thế đỉnh:    tay ngang vai ~80°
        # → Đếm khi góc TĂNG (invert=True)
        "joints": [
            (23, 11, 13),   # hông trái  → vai trái  → khuỷu trái
            (24, 12, 14),   # hông phải  → vai phải  → khuỷu phải
        ],
        "down_angle": 20,
        "up_angle":   75,
        "invert":     True,
        "mode":       "both",
    },

    "high_knees": {
        # Khớp: vai → hông → gối  →  đo góc nâng đầu gối
        # Tư thế đứng:   góc lớn ~160°
        # Tư thế gối lên: góc nhỏ ~85°
        # → Đếm khi góc GIẢM (invert=False), mode=any (chân xen kẽ)
        "joints": [
            (11, 23, 25),   # vai trái  → hông trái  → gối trái
            (12, 24, 26),   # vai phải  → hông phải  → gối phải
        ],
        "down_angle": 155,
        "up_angle":   90,
        "invert":     False,
        "mode":       "any",   # từng chân xen kẽ
    },

    "jumping_jack": {
        # Khớp: hông → vai → khuỷu  →  đo góc dang tay (giống lateral raise)
        # Tư thế bắt đầu: tay thả ~20°
        # Tư thế đỉnh:    tay lên cao ~85°
        # → Đếm khi góc TĂNG (invert=True), mode=both (2 tay cùng lúc)
        "joints": [
            (23, 11, 13),
            (24, 12, 14),
        ],
        "down_angle": 20,
        "up_angle":   80,
        "invert":     True,
        "mode":       "both",
    },
}

DEFAULT_TARGET_REPS = 12
DEFAULT_TARGET_SETS = 3
STABILITY_FRAMES    = 30
