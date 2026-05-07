import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import IMAGE_SIZE, MEAN, STD, DATA_DIR, BATCH_SIZE, TEST_SIZE, VAL_SIZE, RANDOM_STATE

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

def get_dataloaders():
    train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    val_test_dataset = datasets.ImageFolder(DATA_DIR, transform=val_test_transforms)
    indices = np.arange(len(train_dataset.samples))
    labels  = train_dataset.targets

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels,
        test_size=(TEST_SIZE + VAL_SIZE),
        stratify=labels,
        random_state=RANDOM_STATE
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=RANDOM_STATE
    )

    train_data = Subset(train_dataset, train_idx)
    val_data = Subset(val_test_dataset, val_idx)
    test_data = Subset(val_test_dataset, test_idx)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader, val_test_dataset