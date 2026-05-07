import os
import torch
import torch.nn as nn
import mlflow.pytorch

from src.config import ( 
    MODELS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, BASELINE_LR, BASELINE_EPOCHS, BASELINE_PATIENCE, 
    EFF_HEAD_LR, EFF_FULL_LR, EFF_HEAD_EPOCHS, EFF_FULL_EPOCHS, EFF_HEAD_PATIENCE, EFF_FULL_PATIENCE
)
from src.dataset import get_dataloaders
from src.model import ConvNet, build_efficientnet
from src.train import train, evaluate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_loader, val_loader, test_loader, dataset = get_dataloaders()
    num_classes = len(dataset.classes)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Baseline CNN
    with mlflow.start_run(run_name="baseline_cnn"):
        mlflow.log_params({
            "model": "ConvNet",
            "lr": BASELINE_LR,
            "epochs": BASELINE_EPOCHS,
            "patience": BASELINE_PATIENCE,
            "batch_size": train_loader.batch_size,
        })

        baseline = ConvNet(num_classes).to(device)
        optimizer = torch.optim.Adam(baseline.parameters(), lr=BASELINE_LR)
        baseline = train(
            baseline, train_loader, val_loader, criterion, optimizer,
            BASELINE_EPOCHS, BASELINE_PATIENCE, device, "Baseline_CNN"
        )

        acc, f1 = evaluate(baseline, test_loader, device)
        mlflow.log_metrics({"test_accuracy": acc, "test_macro_f1": f1})
        mlflow.pytorch.log_model(baseline, "model")

        torch.save(baseline.state_dict(), os.path.join(MODELS_DIR, "baseline_cnn.pth"))
        print(f"\nBaseline Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")

    # EfficientNet-B0
    with mlflow.start_run(run_name="efficientNet_b0"):
        mlflow.log_params({
            "model": "EfficientNet-B0",
            "head_lr": EFF_HEAD_LR,
            "full_lr": EFF_FULL_LR,
            "head_epochs": EFF_HEAD_EPOCHS,
            "full_epochs": EFF_FULL_EPOCHS,
            "batch_size": train_loader.batch_size,
        })

        eff = build_efficientnet(num_classes).to(device)

        # Phase 1: head only
        opt1 = torch.optim.Adam(
            filter(lambda p: p.requires_grad, eff.parameters()), lr=EFF_HEAD_LR
        )
        eff = train(
            eff, train_loader, val_loader, criterion, opt1,
            EFF_HEAD_EPOCHS, EFF_HEAD_PATIENCE, device, "Phase_1_Head"
        )

        # Phase 2: full fine-tune
        for param in eff.parameters():
            param.requires_grad = True
        opt2 = torch.optim.Adam(eff.parameters(), lr=EFF_FULL_LR)
        eff = train(
            eff, train_loader, val_loader, criterion, opt2,
            EFF_FULL_EPOCHS, EFF_FULL_PATIENCE, device, "Phase_2_Full"
        )

        acc, f1 = evaluate(eff, test_loader, device)
        mlflow.log_metrics({"test_accuracy": acc, "test_macro_f1": f1})
        mlflow.pytorch.log_model(eff, "model")

        torch.save(eff.state_dict(), os.path.join(MODELS_DIR, "efficientNet_b0.pth"))
        print(f"\nEfficientNet Accuracy: {acc:.4f},  Macro F1: {f1:.4f}")


if __name__ == "__main__":
    main()