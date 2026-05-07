<div align="center">

# 🏋️ AI Trainer

**Ứng dụng nhận diện và đếm số lần tập thể dục bằng webcam**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.35-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

## Tính năng

- 🎯 Nhận diện **7 bài tập**: Bicep Curl, Squat, Push-up, Shoulder Press, Lateral Raise, High Knees, Jumping Jack
- 🔢 Đếm rep tự động bằng **State Machine** (tính góc khớp)
- 🤖 Nhận diện tên bài tập bằng **SVM** (Machine Learning)
- 📋 **Workout Plan** — tập theo combo bài thiết kế sẵn
- 🏆 Màn hình **chúc mừng** khi hoàn thành workout
- 📊 HUD hiển thị góc, rep, set, progress bar

---

## Yêu cầu hệ thống

| Thứ | Yêu cầu |
|-----|---------|
| Python | **3.10** (bắt buộc — 3.11+ không tương thích mediapipe) |
| Webcam | Bất kỳ (720p trở lên khuyến nghị) |
| RAM | 4GB+ |
| OS | Windows / macOS / Linux |

---

## Cài đặt

### Bước 1 — Clone repo

```bash
git clone https://github.com/<your-username>/ai_trainer.git
cd ai_trainer
```

### Bước 2 — Cài thư viện

```bash
pip install opencv-python mediapipe==0.10.35 numpy scikit-learn pandas
```

> ⚠️ Nếu đang dùng Python 3.11+, tạo môi trường Python 3.10 trước:
> ```bash
> conda create -n ai_trainer python=3.10 -y
> conda activate ai_trainer
> pip install opencv-python mediapipe==0.10.35 numpy scikit-learn pandas
> ```

### Bước 3 — Tải model MediaPipe

Tải file và đặt vào thư mục `models/`:

```bash
# Windows (PowerShell)
curl -o models/pose_landmarker_full.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

# macOS / Linux
wget -O models/pose_landmarker_full.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
```

Hoặc tải thủ công tại link trên, đặt vào `models/pose_landmarker_full.task`.

---

## Chạy nhanh (không cần train)

Nếu repo đã có sẵn file `models/svm_model.pkl`, chạy thẳng:

```bash
python main.py
```

Chọn workout plan bằng phím `1` `2` `3` `4` và bắt đầu tập.

---

## Train lại model SVM (tùy chọn)

Thực hiện khi muốn thêm dữ liệu mới hoặc cải thiện độ chính xác.

### Bước 1 — Chuẩn bị video

Tạo thư mục và đặt video vào đúng chỗ:

```
data/videos/
├── bicep_curl/        ← 2-3 video quay động tác curl
├── squat/
├── pushup/
├── shoulder_press/
├── lateral_raise/
├── high_knees/
├── jumping_jack/
└── idle/              ← video đứng im, không tập
```

**Yêu cầu video:**
- Định dạng: `.mp4` `.avi` `.mov`
- Thời lượng: 30–60 giây mỗi video
- Góc quay: thấy toàn thân, cách camera 1.5–2m
- Số lượng: tối thiểu 2 video mỗi bài

### Bước 2 — Tạo dataset

```bash
python build_dataset.py
```

→ Sinh ra `data/dataset.csv`

### Bước 3 — Train SVM

```bash
python train_svm.py
```

→ In ra **Accuracy** và **Confusion Matrix**
→ Sinh ra `models/svm_model.pkl`

### Bước 4 — Chạy app

```bash
python main.py
```

---

## Cấu trúc project

```
ai_trainer/
├── main.py              ← Vòng lặp chính + UI
├── config.py            ← Toàn bộ hằng số (chỉnh ở đây)
├── pose_extractor.py    ← MediaPipe wrapper (trích xuất + chuẩn hóa)
├── exercise_logic.py    ← State machine đếm rep + session manager
├── workout_plans.py     ← Định nghĩa combo bài tập
├── build_dataset.py     ← Video → CSV
├── train_svm.py         ← Train SVM → pkl
├── models/
│   ├── pose_landmarker_full.task   ← Tải từ Google (xem hướng dẫn trên)
│   └── svm_model.pkl               ← Tự sinh sau train
└── data/
    ├── dataset.csv                 ← Tự sinh sau build_dataset
    └── videos/
        └── <tên_bài>/              ← Đặt video vào đây
```

---

## Phím tắt

| Phím | Tác dụng |
|------|----------|
| `1` `2` `3` `4` | Chọn workout plan |
| `ESC` | Về màn hình chọn plan |
| `R` | Restart plan hiện tại |
| `Q` | Thoát |

---

## Tùy chỉnh

### Đổi kích thước cửa sổ, mục tiêu rep/set

Mở `config.py`:

```python
WINDOW_W            = 1280   # chiều rộng cửa sổ
WINDOW_H            = 720
DEFAULT_TARGET_REPS = 12
DEFAULT_TARGET_SETS = 3
STABILITY_FRAMES    = 30     # tăng nếu hay bị lẫn bài tập
```

### Thêm workout plan mới

Mở `workout_plans.py`, thêm vào `WORKOUT_PLANS`:

```python
"5": {
    "name":      "Tên combo",
    "exercises": [
        {"exercise": "squat",      "reps": 15, "sets": 3},
        {"exercise": "high_knees", "reps": 20, "sets": 2},
    ],
},
```

### Chỉnh ngưỡng góc bài tập

Mở `config.py`, tìm `EXERCISE_CONFIG`, sửa `up_angle` / `down_angle` của bài cần chỉnh.

---

## Công nghệ sử dụng

| Thành phần | Công nghệ | Vai trò |
|-----------|-----------|---------|
| Pose Estimation | MediaPipe BlazePose | Phát hiện 33 điểm khớp cơ thể |
| Phân loại bài tập | SVM (scikit-learn) | Nhận diện tên bài đang tập |
| Đếm rep | State Machine (toán học) | Đếm số lần dựa trên góc khớp |
| Giao diện | OpenCV | Vẽ HUD, skeleton, progress bar |

---

## Troubleshooting

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
→ Đang dùng mediapipe sai phiên bản. Chạy: `pip install mediapipe==0.10.35`

**`FileNotFoundError: pose_landmarker_full.task`**
→ Chưa tải model. Làm theo Bước 3 trong phần Cài đặt.

**Webcam không mở được**
→ Đổi số `0` thành `1` hoặc `2` trong `cap = cv2.VideoCapture(0)` ở `main.py`.

**Đếm rep không chính xác**
→ Đứng xa camera hơn (1.5–2m), đảm bảo thấy toàn thân. Thử chỉnh `up_angle`/`down_angle` trong `config.py`.

---

## License

MIT License — tự do sử dụng, chỉnh sửa, chia sẻ.
