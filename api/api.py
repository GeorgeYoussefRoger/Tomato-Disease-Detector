from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import numpy as np

from src.config import CLASS_NAMES, MEAN, STD, MODELS_DIR, DROPOUT
from src.dataset import val_test_transforms
from src.model import ConvNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet(num_classes=len(CLASS_NAMES), dropout=DROPOUT)
model.load_state_dict(torch.load(f"{MODELS_DIR}/best_model.pth", map_location=device))
model.to(device).eval()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = val_test_transforms(img).unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            out = model(img_tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_name = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])
        return {"prediction": pred_name, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}