import torch
import mlflow
import os
from tqdm import tqdm

from src.config import MODELS_DIR

def train(model, train_loader, val_loader, criterion, optimizer, epochs, patience, device):
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(MODELS_DIR, "best_model.pth")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}")
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        val_pbar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{epochs}")
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device)

                outputs  = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)

        mlflow.log_metrics({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }, step=epoch)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    model.load_state_dict(torch.load(best_model_path))
    return model