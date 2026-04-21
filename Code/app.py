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
from gradcam import (GradCAM, get_target_layer, overlay_heatmap,
                     DualInputGradCAM, load_dual_input_model)

# Hugging Face Hub model repo — weights are downloaded automatically
# when running as a deployed Space and local .pth files are absent.
# A Space variable can override this at deploy time.
HF_MODEL_REPO = os.getenv(
    "HF_MODEL_REPO",
    "DhanushGWU1995/facial-pain-detection-models",
)
HF_MODEL_FILES = {
    "custom_cnn":    "custom_cnn_best.pth",
    "vgg16":         "vgg16_best.pth",
    "resnet50":      "resnet50_best.pth",
    "efficientnet":  "efficientnet_best.pth",
    "vgg16_mouth":   "vgg16_mouth_attention_best.pth",
    "resnet50_mouth": "resnet50_mouth_attention_best.pth",
    "dual_input":    "dual_input_best.pth",
}


def _resolve_model_path(key: str) -> str | None:
    """
    Return a path to the model weights file.
    Priority: local models/ folder → Hugging Face Hub download.
    """
    local_path = MODEL_PATHS[key]
    if os.path.exists(local_path):
        return local_path

    # Attempt HF Hub download (only if huggingface_hub is installed)
    try:
        from huggingface_hub import hf_hub_download
        st.info(f"⬇️ Downloading {HF_MODEL_FILES[key]} from Hugging Face Hub…")
        cached = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=HF_MODEL_FILES[key],
        )
        return cached
    except Exception:
        return None

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _build_mouth_attn_model(arch: str):
    """Lazy-import and instantiate a mouth-attention model."""
    import importlib.util
    from pathlib import Path as _Path
    spec = importlib.util.spec_from_file_location(
        "mouth_attention",
        _Path(__file__).parent / "09_finetune_mouth_attention.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = mod.VGG16WithAttention if arch == "vgg16_mouth" else mod.ResNet50WithAttention
    return cls(NUM_CLASSES)


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
    elif model_name == "VGG-16 + Mouth Attention":
        model = _build_mouth_attn_model("vgg16_mouth")
        key = "vgg16_mouth"
    elif model_name == "ResNet-50 + Mouth Attention":
        model = _build_mouth_attn_model("resnet50_mouth")
        key = "resnet50_mouth"
    elif model_name == "Dual-Input CNN+MLP ★★":
        # Loaded separately via load_dual_input_model (two-input architecture)
        key = "dual_input"
        path = _resolve_model_path(key)
        if path is None:
            return None, key
        model = load_dual_input_model(path)
        model.eval()
        return model.to(DEVICE), key
    else:
        model = models.efficientnet_b3(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE), nn.Linear(1536, 512),
            nn.SiLU(True), nn.Dropout(DROPOUT_RATE), nn.Linear(512, NUM_CLASSES)
        )
        key = "efficientnet"

    path = _resolve_model_path(key)
    if path is None:
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


def _mouth_crop_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Crop the lower 45% of a face image (mouth/chin) and return a tensor."""
    w, h = pil_img.size
    top = int(0.55 * h)
    mouth_pil = pil_img.crop((0, top, w, h))
    return PREPROCESS(mouth_pil.resize((IMG_SIZE, IMG_SIZE)))


def predict(model, pil_img: Image.Image, model_key: str = ""):
    """Run inference; handles both single-input and dual-input models."""
    full_tensor = PREPROCESS(pil_img.resize((IMG_SIZE, IMG_SIZE))).unsqueeze(0).to(DEVICE)

    if model_key == "dual_input":
        mouth_tensor = _mouth_crop_tensor(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(full_tensor, mouth_tensor)
    else:
        with torch.no_grad():
            logits = model(full_tensor)

    probs    = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def run_gradcam(model, model_key: str, pil_img: Image.Image):
    """
    Generate Grad-CAM overlay(s).

    For dual-input: returns a dict with 'full' and 'mouth' overlays.
    For all others: returns a single overlay image and CAM array.
    """
    if model_key == "dual_input":
        gcam = DualInputGradCAM(model)
        full_t  = PREPROCESS(pil_img.resize((IMG_SIZE, IMG_SIZE)))
        mouth_t = _mouth_crop_tensor(pil_img)

        # Mouth crop as PIL for display
        w, h = pil_img.size
        mouth_pil = pil_img.crop((0, int(0.55 * h), w, h)).resize((IMG_SIZE, IMG_SIZE))

        full_cam, mouth_cam, _, _ = gcam.generate(full_t, mouth_t)
        full_rgb  = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
        mouth_rgb = np.array(mouth_pil)
        return {
            "full":  overlay_heatmap(full_rgb,  full_cam),
            "mouth": overlay_heatmap(mouth_rgb, mouth_cam),
            "mouth_pil": mouth_pil,
        }

    target_layer = get_target_layer(model_key, model)
    gradcam_obj  = GradCAM(model, target_layer)
    tensor = PREPROCESS(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    cam, _, _ = gradcam_obj.generate(tensor)
    original  = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    overlay   = overlay_heatmap(original, cam)
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
        "A deep learning system for **objective, non-invasive pain monitoring** from facial "
        "expressions — designed to support clinical staff in assessing pain for patients who "
        "cannot self-report (ICU, post-surgical, neonatal, dementia care)."
    )

    st.warning(
        "⚕️ **Clinical Disclaimer:** This tool is a research prototype for educational purposes "
        "only. It is not a certified medical device and must not be used for clinical diagnosis "
        "or treatment decisions without validation by qualified healthcare professionals."
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        model_choice = st.selectbox(
            "Select Model",
            [
                "Dual-Input CNN+MLP ★★",       # full face + mouth crop fusion
                "VGG-16 + Mouth Attention",
                "ResNet-50 + Mouth Attention",
                "VGG-16",
                "ResNet-50",
                "EfficientNet-B3",
                "Custom CNN",
            ],
            index=0,
            help="★★ = dual-input (full face + mouth crop CNN+MLP); "
                 "shows two Grad-CAMs side-by-side",
        )
        show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)
        input_mode = st.radio("Input Mode", ["Webcam", "Upload Image"])

        st.divider()
        st.markdown("### 🏨 Target Use Cases")
        st.markdown(
            "- **ICU patients** on mechanical ventilation\n"
            "- **Post-surgical** recovery monitoring\n"
            "- **Neonatal** pain assessment\n"
            "- **Dementia / non-verbal** patients\n"
            "- **Telemedicine** pain screening"
        )
        st.markdown("*Supplements CPOT and FLACC behavioral scales.*")

    model, model_key = load_model(model_choice)
    if model is None:
        st.error(
            f"Model weights not found locally at `{MODEL_PATHS[model_key]}` "
            f"or in Hugging Face repo `{HF_MODEL_REPO}` "
            f"(expected file: `{HF_MODEL_FILES[model_key]}`). "
            "Upload the checkpoint with `08_upload_to_hf.py` or add the file manually."
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

        pred_idx, probs = predict(model, face_crop, model_key)
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

            gradcam_result = run_gradcam(model, model_key, face_crop)

            if isinstance(gradcam_result, dict):
                # Dual-input model — show both branch heat maps
                st.markdown(
                    "**Dual-branch Grad-CAM**: left column = full-face encoder "
                    "(ResNet-50), right column = mouth-crop encoder (ResNet-18). "
                    "Red = high activation → what each branch focuses on."
                )
                gcol1, gcol2, gcol3, gcol4 = st.columns(4)
                with gcol1:
                    st.image(face_crop.resize((IMG_SIZE, IMG_SIZE)),
                             caption="Full Face", use_column_width=True)
                with gcol2:
                    st.image(gradcam_result["full"],
                             caption="Grad-CAM: Full-Face Branch",
                             use_column_width=True)
                with gcol3:
                    st.image(gradcam_result["mouth_pil"],
                             caption="Mouth Crop", use_column_width=True)
                with gcol4:
                    st.image(gradcam_result["mouth"],
                             caption="Grad-CAM: Mouth Branch",
                             use_column_width=True)
            else:
                overlay_img, _ = gradcam_result
                st.markdown(
                    "The heatmap highlights the **facial regions** (brow, eyes, mouth) "
                    "that most influenced the prediction."
                )
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
