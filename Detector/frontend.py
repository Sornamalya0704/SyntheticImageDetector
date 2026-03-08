import streamlit as st
from PIL import Image
import cv2
import pandas as pd
from app.inference import predict

st.set_page_config(
    page_title="Synthetic Image Detector",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Synthetic Image Detector")

st.sidebar.header("Settings")

mode = st.sidebar.radio(
    "Choose Mode",
    ["Upload Image", "Webcam"]
)

# Prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------
# IMAGE UPLOAD MODE
# --------------------------------

if mode == "Upload Image":

    col1, col2 = st.columns(2)

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        with col1:
            st.image(image, caption="Uploaded Image")

        pred, conf, probs = predict(image)

        label = "REAL" if pred == 0 else "SYNTHETIC"

        with col2:

            st.subheader("Prediction")

            if label == "REAL":
                st.success(f"{label}")
            else:
                st.error(f"{label}")

            st.metric("Confidence", f"{conf*100:.2f}%")

            prob_df = pd.DataFrame(
                {
                    "Class": ["Real", "Synthetic"],
                    "Probability": probs
                }
            )

            st.bar_chart(prob_df.set_index("Class"))

        st.session_state.history.append(label)

# --------------------------------
# WEBCAM MODE
# --------------------------------

elif mode == "Webcam":

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:

        ret, frame = camera.read()

        if not ret:
            st.error("Camera not available")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(frame)

        pred, conf, probs = predict(image)

        label = "REAL" if pred == 0 else "SYNTHETIC"

        color = (0,255,0) if label == "REAL" else (255,0,0)

        cv2.putText(
            frame,
            f"{label} ({conf*100:.1f}%)",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        FRAME_WINDOW.image(frame)

        st.session_state.history.append(label)

    camera.release()

# --------------------------------
# PREDICTION HISTORY
# --------------------------------

st.sidebar.subheader("Prediction History")

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(
        st.session_state.history,
        columns=["Prediction"]
    )

    st.sidebar.dataframe(history_df.tail(10))