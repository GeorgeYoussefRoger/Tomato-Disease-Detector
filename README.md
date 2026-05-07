# 🍅 Tomato Disease Detection

Deep learning pipeline for classifying tomato leaf diseases using the PlantVillage dataset. Compares a custom CNN baseline against a fine-tuned EfficientNet-B0, with Grad-CAM explainability and a Streamlit inference app.

> Trained on 10 Tomato Disease Classes from [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

## 🚀 Features

- Custom CNN baseline trained from scratch
- Two-phase EfficientNet-B0 fine-tuning (frozen backbone -> full fine-tune)
- Grad-CAM heatmaps for prediction explainability
- Experiment tracking with MLflow
- Interactive inference app with Streamlit

## 📦 Installation & Usage

- Prerequisites
  - Python 3.12+

1. Clone the repository

```
git clone https://github.com/GeorgeYoussefRoger/Tomato-Disease-Detector.git
cd Tomato-Disease-Detector
```

2. Create a Virtual Environment

```
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Run UI

```
streamlit run app.py
```

5. Access UI -> http://localhost:8501

## 🧠 Training

- Train Models

```
python -m src.main
```

- View MLflow experiments

```
mlflow server --backend-store-uri sqlite:///mlruns.db
```

## 🤖 Models Details

### Model Architectures

- Baseline CNN
  - 4 Conv Blocks
  - BatchNorm
  - MaxPooling
  - Global Average Pooling
  - Dropout

- EfficientNet-B0
  - ImageNet pretrained
  - Two-stage fine-tuning

### Model Results

| Model           | Accuracy | Macro F1 |
| --------------- | -------- | -------- |
| Baseline CNN    | 95.00%   | 0.93     |
| EfficientNet-B0 | 99.85%   | 0.99     |

- Note: PlantVillage contains controlled-condition images so real-world field performance may differ.

## 📂 Project Structure

```
├── src/                # ML Pipeline
├── app.py              # Streamlit UI
├── models/             # Saved model weights
└── requirements.txt
```

## 📜 License

- This project is licensed under the MIT License.
- See the [LICENSE](LICENSE) file for more details.
