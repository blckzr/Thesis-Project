import streamlit as st
import cv2
from ultralytics import YOLO

def main():
    # Page Setup
    st.set_page_config(page_title="YOLO11Nano Live", layout="wide")
    _ = st.title("YOLO11Nano Real-Time Detection")

    _ = st.sidebar.header("Model Settings")
    conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.4)

    model = YOLO("yolo11n.pt")

    frame_placeholder = st.empty()

    cap = cv2.VideoCapture(0)

    stop_button = st.button("Stop Stream")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.write("Video capture failed.")
            break

        results = model.track(frame, persist=True, conf=conf_threshold)

        annotated_frame = results[0].plot()

        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        _ = frame_placeholder.image(annotated_frame_rgb, channels="RGB")

    cap.release()

if __name__ == "__main__":
    main()
