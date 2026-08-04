import torch
import torch.nn as nn
import mlflow

from src.config import *
from src.dataset import create_dataloaders
from src.model import ConvNet
from src.train import train
from src.evaluate import evaluate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, dataset = create_dataloaders()
    num_classes = len(dataset.classes)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="ConvNet"):
        mlflow.log_params({
            "lr": LEARNING_RATE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
        })

        model = ConvNet(num_classes, DROPOUT).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        model = train(model, train_loader, val_loader, criterion, optimizer, EPOCHS, PATIENCE, device)

        acc, f1 = evaluate(model, test_loader, device)
        mlflow.log_metrics({"test_accuracy": acc, "test_macro_f1": f1})
        print(f"\nAccuracy: {acc:.4f}, Macro F1: {f1:.4f}")

if __name__ == "__main__":
    main()