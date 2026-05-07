# train_svm.py — THÀNH VIÊN 2: AI Engineer (Phần 2)
# Nhiệm vụ: Đọc CSV → train SVM → lưu pkl
#
# Chạy: python train_svm.py

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from config import DATA_CSV, SVM_PATH

os.makedirs(os.path.dirname(SVM_PATH), exist_ok=True)


def train():
    # --- Đọc data ---
    if not os.path.exists(DATA_CSV):
        print(f"[ERROR] Không tìm thấy: {DATA_CSV}")
        print("Hãy chạy build_dataset.py trước.")
        return

    df = pd.read_csv(DATA_CSV)
    print(f"[INFO] Dataset: {len(df)} mẫu, {df['label'].value_counts().to_dict()}")

    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].values

    # --- Chia tập train / test (80/20) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Chuẩn hóa feature (StandardScaler) ---
    # SVM nhạy cảm với scale → cần normalize feature về mean=0, std=1
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # --- Train SVM ---
    print("[INFO] Đang train SVM (kernel=linear)...")
    model = SVC(
        kernel="linear",
        probability=True,   # Bật để lấy confidence score trong main.py
        C=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # --- Đánh giá ---
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n[RESULT] Accuracy: {acc * 100:.1f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # --- Lưu model + scaler vào 1 file pkl ---
    bundle = {"model": model, "scaler": scaler}
    with open(SVM_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n[DONE] Model đã lưu: {SVM_PATH}")


if __name__ == "__main__":
    train()
