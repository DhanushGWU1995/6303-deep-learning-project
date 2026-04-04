"""
app.py — Streamlit Demo App
---------------------------
Real-time facial pain detection via webcam or image upload.
Displays pain level prediction, confidence scores, and Grad-CAM heatmap.

Run:
  streamlit run app.py
"""

import os
import io
import json
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import streamlit as st

from config import (
    IMG_SIZE, NUM_CLASSES, DEVICE, MODEL_PATHS,
    PAIN_LABELS, PAIN_COLORS, CLASSES, DROPOUT_RATE
)
from train_custom_cnn import CustomCNN
from gradcam import GradCAM, get_target_layer, overlay_heatmap

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model(model_name: str):
    if model_name == "Custom CNN":
        model = CustomCNN(NUM_CLASSES)
        key = "custom_cnn"
    elif model_name == "VGG-16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Sequential(
            nn.Dropout(DROPOUT_RATE), nn.Linear(4096, 256),
            nn.ReLU(True), nn.Dropout(DROPOUT_RATE), nn.Linear(256, NUM_CLASSES)
        )
        key = "vgg16"
    elif model_name == "ResNet-50":
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(DROPOUT_RATE), nn.Linear(2048, 512),
            nn.ReLU(True), nn.Dropout(DROPOUT_RATE), nn.Linear(512, NUM_CLASSES)
        )
        key = "resnet50"
    else:
        model = models.efficientnet_b3(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE), nn.Linear(1536, 512),
            nn.SiLU(True), nn.Dropout(DROPOUT_RATE), nn.Linear(512, NUM_CLASSES)
        )
        key = "efficientnet"

    path = MODEL_PATHS[key]
    if not os.path.exists(path):
        return None, key
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE), key


def detect_face(pil_img: Image.Image):
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces  = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if len(faces) == 0:
        return pil_img, None
    x, y, w, h = faces[0]
    face_crop = pil_img.crop((x, y, x + w, y + h))
    return face_crop, (x, y, w, h)


def predict(model, pil_img: Image.Image):
    tensor = PREPROCESS(pil_img.resize((IMG_SIZE, IMG_SIZE))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def run_gradcam(model, model_key, pil_img: Image.Image):
    target_layer = get_target_layer(model_key, model)
    gradcam = GradCAM(model, target_layer)
    tensor = PREPROCESS(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    cam, pred_class, confidence = gradcam.generate(tensor)
    original = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    overlay  = overlay_heatmap(original, cam)
    return overlay, cam


def build_report(pain_label, confidence, probs):
    report = {
        "pain_level": pain_label,
        "confidence": f"{confidence * 100:.1f}%",
        "all_probabilities": {
            PAIN_LABELS[i]: f"{p * 100:.1f}%" for i, p in enumerate(probs)
        },
    }
    return json.dumps(report, indent=2)


def main():
    st.set_page_config(
        page_title="Facial Pain Detection",
        page_icon="🏥",
        layout="wide",
    )

    st.title("🏥 Real-Time Facial Pain Detection")
    st.markdown(
        "Upload an image or use your **webcam** to detect pain levels from facial expressions. "
        "Powered by deep learning with Grad-CAM interpretability."
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        model_choice = st.selectbox(
            "Select Model",
            ["EfficientNet-B3", "ResNet-50", "VGG-16", "Custom CNN"],
            index=0,
        )
        show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)
        input_mode = st.radio("Input Mode", ["Webcam", "Upload Image"])

    model, model_key = load_model(model_choice)
    if model is None:
        st.error(
            f"Model weights not found at `{MODEL_PATHS[model_key]}`. "
            "Please train the model first by running the training scripts."
        )
        return

    st.sidebar.success(f"✅ {model_choice} loaded")
    st.sidebar.info(f"Device: {DEVICE.upper()}")

    image = None

    if input_mode == "Webcam":
        st.subheader("📸 Webcam Capture")
        cam_img = st.camera_input("Take a photo of your face")
        if cam_img:
            image = Image.open(cam_img).convert("RGB")
    else:
        st.subheader("📂 Upload Image")
        uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")

    if image is not None:
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            face_crop, bbox = detect_face(image)

            if bbox is not None:
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                x, y, w, h = bbox
                cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
                annotated = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                st.image(annotated, caption="Face Detected", use_column_width=True)
            else:
                st.image(image, caption="No face detected — using full image",
                         use_column_width=True)

        pred_idx, probs = predict(model, face_crop)
        pain_label  = PAIN_LABELS[pred_idx]
        confidence  = float(probs[pred_idx])
        pain_color  = PAIN_COLORS[pred_idx]

        with col2:
            st.subheader("🔍 Analysis Results")
            st.markdown(
                f"<h2 style='color:{pain_color};'>{pain_label}</h2>",
                unsafe_allow_html=True,
            )
            st.metric("Confidence", f"{confidence * 100:.1f}%")

            st.subheader("Class Probabilities")
            for i, cls in enumerate(CLASSES):
                st.progress(float(probs[i]),
                            text=f"{PAIN_LABELS[i]}: {probs[i]*100:.1f}%")

        if show_gradcam:
            st.divider()
            st.subheader("🔥 Grad-CAM Interpretability")
            st.markdown(
                "The heatmap highlights the **facial regions** (brow, eyes, mouth) "
                "that most influenced the prediction."
            )
            overlay_img, _ = run_gradcam(model, model_key, face_crop)
            gcol1, gcol2 = st.columns(2)
            with gcol1:
                st.image(face_crop.resize((IMG_SIZE, IMG_SIZE)),
                         caption="Cropped Face", use_column_width=True)
            with gcol2:
                st.image(overlay_img,
                         caption="Grad-CAM Overlay", use_column_width=True)

        st.divider()
        st.subheader("📋 Report")
        report_json = build_report(pain_label, confidence, probs)
        st.code(report_json, language="json")

        report_bytes = report_json.encode("utf-8")
        st.download_button(
            label="⬇️ Download Report (JSON)",
            data=report_bytes,
            file_name="pain_detection_report.json",
            mime="application/json",
        )

    st.divider()
    st.caption(
        "DATS 6303 — Deep Learning Final Project | "
        "George Washington University | "
        "Facial Pain Detection using CNN Transfer Learning + Grad-CAM"
    )


if __name__ == "__main__":
    main()
