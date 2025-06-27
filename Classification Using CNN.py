import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


class MedicalImageDataset(Dataset):
    """Dataset for loading CT volumes and corresponding RT structures."""
    def __init__(self, root_dir: str):
        self.ct_dir = os.path.join(root_dir, "ct")
        self.rt_dir = os.path.join(root_dir, "rt")

        if not os.path.isdir(self.ct_dir):
            raise FileNotFoundError(f"CT directory not found: {self.ct_dir}")
        if not os.path.isdir(self.rt_dir):
            raise FileNotFoundError(f"RT directory not found: {self.rt_dir}")

        self.ct_files = sorted(glob.glob(os.path.join(self.ct_dir, "*.nii")))
        self.rt_files = sorted(glob.glob(os.path.join(self.rt_dir, "*.nii")))

        if len(self.ct_files) == 0:
            raise FileNotFoundError(
                f"No NIfTI files with extension '.nii' found in {self.ct_dir}"
            )
        if len(self.rt_files) == 0:
            raise FileNotFoundError(
                f"No NIfTI files with extension '.nii' found in {self.rt_dir}"
            )
        if len(self.ct_files) != len(self.rt_files):
            raise ValueError(
                "Number of CT and RT files must match"
            )

    def __len__(self) -> int:
        return len(self.ct_files)

    def __getitem__(self, idx: int):
        ct_path = self.ct_files[idx]
        rt_path = self.rt_files[idx]
        ct_volume = nib.load(ct_path).get_fdata()
        rt_volume = nib.load(rt_path).get_fdata()
        ct_tensor = torch.tensor(ct_volume, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(int(rt_volume.sum() > 0), dtype=torch.long)
        return ct_tensor, label


def load_dataset_nifti(path: str):
    dataset = MedicalImageDataset(path)
    if len(dataset) == 0:
        raise ValueError(f"Dataset at {path} is empty")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class CNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
            nn.AdaptiveAvgPool3d(1)
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def performance_estimate(dataset, model, loss_fn, device, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    num_batches = len(dataloader)
    size = len(dataset)
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
    return correct / size, loss / num_batches


def one_epoch(model, loss_fn, optimizer, dataset, device, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def training(datasets_split, device, batch_size=1, lr=1e-3, weight_decay=1e-8, epochs=1):
    train_dataset, test_dataset = datasets_split
    model = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        one_epoch(model, loss_fn, optimizer, train_dataset, device, batch_size)
        acc_train, _ = performance_estimate(train_dataset, model, loss_fn, device, batch_size)
        acc_test, _ = performance_estimate(test_dataset, model, loss_fn, device, batch_size)
        print(f"Epoch {epoch+1}: train acc={acc_train:.3f} test acc={acc_test:.3f}")

    return model


if __name__ == "__main__":
    data_path = r"E:\\test1"
    device = get_device()
    print(f"Using {device} device")
    datasets_split = load_dataset_nifti(data_path)
    model = training(datasets_split, device)
    print("~~~~~ training is done ~~~~~")
