import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from pose_extractor import extract_pose_features

X = []  # fitur pose
y = []  # label

for folder in ["thinking", "idle", "idea", "shocked","hidup_jokowi","monkey_fuck", "other"]:
    if folder == "thinking":
        label = 1
    elif folder == "idle":
        label = 2
    elif folder == "idea":
        label = 3
    elif folder == "shocked":
        label = 4
    elif folder == "hidup_jokowi":
        label = 5
    elif folder == "monkey_fuck":
       label = 6
    else:
        label = 0

    folder_path = os.path.join("images", folder)
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            features = extract_pose_features(img_path)
            if features is not None:
                X.append(features)
                y.append(label)

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print(" Tidak ada data untuk dilatih. Pastikan kamu punya gambar di folder.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Akurasi:", model.score(X_test, y_test))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Simpan model
    joblib.dump(model, "pose_classifier.pkl")
    print("\n Model disimpan sebagai 'pose_classifier.pkl'")