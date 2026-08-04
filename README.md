# 🍅 Tomato Disease Detection

An end-to-end Deep Learning Project that detects tomato leaf diseases with a PyTorch CNN and deployment through a FastAPI backend and Streamlit interface.

> Trained on 10 Tomato Disease Classes from [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

## 🚀 Features

- PyTorch custom CNN trained from scratch
- MLflow experiment tracking
- FastAPI inference API
- Streamlit interactive frontend

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

- Install PyTorch (choose your CUDA version at https://pytorch.org/get-started/locally/)

```
pip install -r requirements.txt
```

4. Run API

```
uvicorn api.api:app
```

5. Run UI

```
streamlit run ui/app.py
```

6. Access:
   - UI -> http://localhost:8501
   - API Docs -> http://localhost:8000/docs

## 🧠 Training

- Train Models

```
python -m src.main
```

- View MLflow experiments

```
mlflow server --backend-store-uri sqlite:///mlruns.db
```

## 🤖 Model Details

- Model Architecture
  - 4 Conv Blocks
  - Batch Normalization
  - Max Pooling
  - Global Average Pooling
  - Dropout

- Model Results
  - Accuracy: 95%
  - Macro F1: 93%

- Note: PlantVillage contains controlled-condition images so real-world field performance may differ.

## 📜 License

This project is licensed under the MIT License.
