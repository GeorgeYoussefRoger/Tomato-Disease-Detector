DATA_DIR = "data"
MODELS_DIR = "models"
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
MLFLOW_EXPERIMENT = "Tomato-Disease-Classification"

CLASS_NAMES = [
    "Bacterial Spot", "Early Blight", "Late Blight",
    "Leaf Mold", "Septoria Leaf Spot", "Spider Mites", 
    "Target Spot", "Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy"
]

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
BATCH_SIZE = 32

LEARNING_RATE = 0.001
EPOCHS = 20
PATIENCE = 5
DROPOUT = 0.5