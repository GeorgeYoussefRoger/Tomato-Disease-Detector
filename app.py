import os
import numpy as np
import torch
from PIL import Image
import streamlit as st
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config import MEAN, STD, MODELS_DIR
from src.dataset import val_test_transforms
from src.model import build_efficientnet

CLASS_NAMES = [
    "Bacterial Spot", "Early Blight", "Late Blight",
    "Leaf Mold", "Septoria Leaf Spot", "Spider Mites", 
    "Target Spot", "Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy"
]

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_efficientnet(num_classes=len(CLASS_NAMES), freeze_backbone=False)
    weights_path = os.path.join(MODELS_DIR, "efficientNet_b0.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model, device

def predict(model, device, img_tensor):
    with torch.no_grad():
        out = model(img_tensor.to(device))
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    return probs

def run_gradcam(model, device, img_tensor, class_idx, img_np):
    target_layers = [model.features[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=img_tensor.to(device), targets=[ClassifierOutputTarget(class_idx)])[0]  
    return show_cam_on_image(img_np, grayscale_cam, use_rgb=True)


st.title("🍅 Tomato Disease Detector")
st.markdown("Upload a tomato leaf image to detect disease with Grad-CAM explainability.")

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    with st.spinner("Analyzing..."):
        model, device = load_model()
        img_tensor = val_test_transforms(image).unsqueeze(0)

        # Denormalize for display
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * np.array(STD) + np.array(MEAN), 0, 1).astype(np.float32)

        probs = predict(model, device, img_tensor)
        pred_idx = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx]

        overlay = run_gradcam(model, device, img_tensor, pred_idx, img_np)

    col1, col2 = st.columns(2)
    col1.image(image, caption="Uploaded image", width=300)
    col2.image(overlay, caption="Grad-CAM — regions influencing the prediction", width=300)
    st.markdown("---")
    col3, col4 = st.columns(2)
    col3.metric("Prediction", pred_name)
    col4.metric("Confidence", f"{confidence:.1%}")

    st.markdown("### Class probabilities")
    for name, prob in sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1]):
        st.progress(float(prob), text=f"{name}: {prob:.1%}")