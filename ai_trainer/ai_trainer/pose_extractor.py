# pose_extractor.py — THÀNH VIÊN 1: Core Dev
# Fix: bỏ type hint tuple[int,int] gây lỗi Pylance trên Python 3.10

from __future__ import annotations   # ← fix Pylance "Variable not allowed in type expression"

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from config import MODEL_PATH, SMOOTH_WINDOW


def _build_landmarker(mode: str) -> vision.PoseLandmarker:
    running_mode = (vision.RunningMode.VIDEO
                    if mode == "VIDEO"
                    else vision.RunningMode.IMAGE)
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


class PoseExtractor:
    """
    Pipeline: frame BGR → (feature_vector 132 chiều, landmark_list).
    mode='VIDEO' cho webcam/video tuần tự (cần timestamp tăng dần).
    mode='IMAGE' cho ảnh đơn lẻ.
    """

    def __init__(self, mode: str = "VIDEO"):
        self.mode       = mode
        self.landmarker = _build_landmarker(mode)
        self._buffer    = deque(maxlen=SMOOTH_WINDOW)
        self._ts        = 0

    # ── PUBLIC ──────────────────────────────────────────────────────────

    def get_features(self, frame_bgr, timestamp_ms: int = None):
        """Trả về (np.ndarray shape 132, landmark_list) hoặc (None, None)."""
        raw_arr, lm_list = self._extract(frame_bgr, timestamp_ms)
        if raw_arr is None:
            return None, None
        smoothed   = self._smooth(raw_arr)
        normalized = self._normalize(smoothed)
        return normalized.flatten().astype(np.float32), lm_list

    def close(self):
        self.landmarker.close()

    # ── PRIVATE ─────────────────────────────────────────────────────────

    def _extract(self, frame_bgr, timestamp_ms):
        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.mode == "VIDEO":
            if timestamp_ms is None:
                self._ts    += 33
                timestamp_ms = self._ts
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None, None

        lm  = result.pose_landmarks[0]
        arr = np.array([[p.x, p.y, p.z, p.visibility] for p in lm],
                       dtype=np.float32)
        return arr, lm

    def _smooth(self, arr: np.ndarray) -> np.ndarray:
        """Moving Average — giảm rung giật khung xương."""
        self._buffer.append(arr)
        return np.mean(self._buffer, axis=0)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa tọa độ:
          Bước 1 — Dời gốc về mid_hip (index 23, 24).
          Bước 2 — Chia cho khoảng cách mid_shoulder → mid_hip
                   → loại bỏ yếu tố xa/gần camera.
        Visibility (cột 3) giữ nguyên.
        """
        lm = arr.copy()

        mid_hip      = (lm[23, :3] + lm[24, :3]) / 2.0
        lm[:, :3]   -= mid_hip

        mid_shoulder = (lm[11, :3] + lm[12, :3]) / 2.0
        scale        = np.linalg.norm(mid_shoulder) + 1e-6
        lm[:, :3]   /= scale

        return lm
