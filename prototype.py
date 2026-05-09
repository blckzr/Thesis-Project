import streamlit as st
import cv2
from ultralytics import YOLO
import os

# uses streamlit and ultralytics to create a fast method to test models.

trained_model_path = "runs/detect/deepfake_detector/weights/best.pt"

def main():
    # Page Setup
    st.set_page_config(page_title="YOLO11Nano Live", layout="wide")
    _ = st.title("YOLO11Nano Real-Time Detection")

    _ = st.sidebar.header("Model Settings")
    conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.4)

    if not os.path.isfile(trained_model_path):
        print("==== WARNING =====")
        print("The trained model does not exist yet. Please train it by running uv run scripts/dataset_train_yolon.py")
        return;

    yolo_model = YOLO(trained_model_path)

    frame_placeholder = st.empty()
    stop_button = st.button("Stop Stream")

    capture = cv2.VideoCapture(0)

    while capture.isOpened() and not stop_button:
        ret, frame = capture.read()
        if not ret:
            st.write("Video capture failed.")
            break

        results = yolo_model.track(frame, persist=True, conf=conf_threshold)

        annotated_frame = results[0].plot()

        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        _ = frame_placeholder.image(annotated_frame_rgb, channels="RGB")

    capture.release()

if __name__ == "__main__":
    main()
