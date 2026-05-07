# build_dataset.py — THÀNH VIÊN 2: AI Engineer (Phần 1)
# Nhiệm vụ: Đọc video từ folder → trích features → lưu CSV
#
# Cấu trúc folder video cần chuẩn bị:
#   data/videos/
#       bicep_curl/   ← chứa các file .mp4/.avi quay bicep curl
#       squat/
#       pushup/
#       idle/         ← chứa video đứng im không tập
#
# Chạy: python build_dataset.py

import os
import cv2
import csv
import numpy as np
from pose_extractor import PoseExtractor
from config import VIDEO_DIR, DATA_CSV, LABELS

os.makedirs(os.path.dirname(DATA_CSV), exist_ok=True)

# Header CSV: feature_0 … feature_131, label
HEADER = [f"f{i}" for i in range(132)] + ["label"]


def process_video(video_path: str, label: str, extractor: PoseExtractor) -> list:
    """Đọc 1 video, trả về list các feature rows."""
    cap  = cv2.VideoCapture(video_path)
    rows = []
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(frame_idx / fps * 1000)
        features, _  = extractor.get_features(frame, timestamp_ms)
        frame_idx   += 1

        if features is None:
            continue

        rows.append(list(features) + [label])

    cap.release()
    return rows


def build_dataset():
    if not os.path.isdir(VIDEO_DIR):
        print(f"[ERROR] Không tìm thấy thư mục video: {VIDEO_DIR}")
        print("Hãy tạo cấu trúc: data/videos/<label>/<video.mp4>")
        return

    total = 0

    with open(DATA_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        for label in LABELS:
            label_dir = os.path.join(VIDEO_DIR, label)
            if not os.path.isdir(label_dir):
                print(f"[SKIP] Không có folder: {label_dir}")
                continue

            video_files = [
                v for v in os.listdir(label_dir)
                if v.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
            ]

            for vf in video_files:
                vpath = os.path.join(label_dir, vf)
                print(f"[INFO] Đang xử lý: {vpath} (label={label})")

                # Tạo extractor mới cho mỗi video (reset timestamp)
                extractor = PoseExtractor(mode="VIDEO")
                rows      = process_video(vpath, label, extractor)
                extractor.close()

                writer.writerows(rows)
                total += len(rows)
                print(f"       → {len(rows)} frames")

    print(f"\n[DONE] Đã lưu {total} mẫu vào: {DATA_CSV}")


if __name__ == "__main__":
    build_dataset()
