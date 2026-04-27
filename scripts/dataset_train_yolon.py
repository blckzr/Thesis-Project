from typing import Literal
import kagglehub
import os
import cv2
import face_recognition
import shutil
import random
import pandas as pd
from ultralytics import YOLO

# =========================================== #
# This file grabs the limited dataset from kaggle and trains it.
#
# The processed dataset files are located in the aptly named "dataset/" folder
# The final model will be in "runs/detect/deepfake_detector/weights/best.pt"
# ============ DATA PREPARATION ============= #


def extract_face_frames(video_path: str, output_dir: str, label: Literal["real"] | Literal["fake"], frame_interval: int =10):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    saved = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)

            for (top, right, bottom, left) in locations:
                face_crop = frame[top:bottom, left:right]
                cv2.imwrite(f"{output_dir}/{label}_{saved:05d}.jpg", face_crop)
                saved += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {saved} face crops from {video_path}")

def generate_labels(images_dir: str, labels_dir: str, class_id: int):
    """Generate YOLO labels: full image bbox for face crops"""
    os.makedirs(labels_dir, exist_ok=True)

    for img_file in os.listdir(images_dir):
        if img_file.endswith(('.jpg', '.png')):
            label_file = os.path.splitext(img_file)[0] + ".txt"
            with open(os.path.join(labels_dir, label_file), "w") as f:
                # class cx cy w h (normalized: full image)
                _ = f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                
def split_train_val(images_dir: str, labels_dir: str, val_ratio: float = 0.2):
    for label in ["real", "fake"]:
        img_src = f"{images_dir}/train/{label}"
        files = [f for f in os.listdir(img_src) if f.endswith(".jpg")]
        val_files = random.sample(files, int(len(files) * val_ratio))

        os.makedirs(f"{images_dir}/val/{label}", exist_ok=True)
        os.makedirs(f"{labels_dir}/val/{label}", exist_ok=True)

        for fname in val_files:
            shutil.move(f"{img_src}/{fname}", f"{images_dir}/val/{label}/{fname}")

            label_fname = fname.replace(".jpg", ".txt")
            shutil.move(
                f"{labels_dir}/train/{label}/{label_fname}",  # ← from label subfolder
                f"{labels_dir}/val/{label}/{label_fname}"     # ← to label subfolder
            )

# ========================================== #

path = kagglehub.dataset_download('simongraves/deepfake-dataset')
df = pd.read_csv(path + "/DeepFake Videos Dataset.csv")

for index, row in df.iterrows():
    original_video = f"{path}/{row["video"]}"
    deepfake_video = f"{path}/{row["deepfake"]}"
    deepfake_image = f"{path}/{row["image"]}"

    print(f"DETECTING FACES FROM {original_video} and {deepfake_video}")

    extract_face_frames(original_video, "dataset/images/train/real", label="real")
    extract_face_frames(deepfake_video, "dataset/images/train/fake", label="fake")

generate_labels("dataset/images/train/real", "dataset/labels/train/real", class_id=0)
generate_labels("dataset/images/train/fake", "dataset/labels/train/fake", class_id=1)

split_train_val("dataset/images", "dataset/labels")


model = YOLO("yolo11n.pt")
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=224,
    batch=32,
    name="deepfake_detector"
)
model.export(format="onnx")


