# AI TRAINER — HƯỚNG DẪN CHẠY

## Cấu trúc project
```
ai_trainer/
├── main.py              ← Chạy chính (Role 4)
├── config.py            ← Cấu hình chung
├── pose_extractor.py    ← Lấy tọa độ, chuẩn hóa (Role 1)
├── exercise_logic.py    ← Đếm rep, quản lý session (Role 3)
├── build_dataset.py     ← Thu thập data (Role 2 - phần 1)
├── train_svm.py         ← Train model (Role 2 - phần 2)
├── models/
│   ├── pose_landmarker_full.task   ← Tải về từ Google
│   └── svm_model.pkl               ← Tự sinh sau khi train
└── data/
    ├── dataset.csv                 ← Tự sinh sau build_dataset
    └── videos/
        ├── bicep_curl/   ← Đặt video vào đây
        ├── squat/
        ├── pushup/
        └── idle/
```

## Bước 0: Cài thư viện
```
pip install opencv-python mediapipe==0.10.35 numpy scikit-learn pandas
```

## Bước 1: Tải model MediaPipe
Tải file này về, đặt vào thư mục models/:
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

## Bước 2: Chuẩn bị video
Quay video mỗi bài tập (30–60 giây mỗi video, ít nhất 2–3 video/bài).
Đặt vào đúng thư mục: data/videos/<tên_bài>/<video.mp4>

Ví dụ:
- data/videos/squat/squat_01.mp4
- data/videos/bicep_curl/curl_01.mp4
- data/videos/pushup/push_01.mp4
- data/videos/idle/standing_01.mp4

## Bước 3: Build dataset
```
python build_dataset.py
```
→ Sinh ra data/dataset.csv

## Bước 4: Train SVM
```
python train_svm.py
```
→ Sinh ra models/svm_model.pkl
→ In ra Accuracy + Confusion Matrix

## Bước 5: Chạy ứng dụng
```
python main.py
```

## Phím tắt khi chạy
- Q : Thoát
- R : Reset session (về hiệp 1, rep 0)
