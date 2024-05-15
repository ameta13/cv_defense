import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def evaluate(model, dataloader: DataLoader, device: str):
    y_true = np.array([])
    y_pred = np.array([])
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            y_true = np.append(y_true, labels.cpu().numpy())
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)
            y_pred = np.append(y_pred, predicted.cpu().numpy())

    return y_true, y_pred

def train_model(model, data_loader: DataLoader, device: str, learning_rate: float = 0.001, num_epochs: int = 5):
    print('Start training')
    start = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    assert data_loader.batch_size >= 2, f'Expected batch_size >= 2, got {data_loader.batch_size}'

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in data_loader:
            if images.shape[0] < 2:
                print(f'train: skipped small batch size - {images.shape=}')
                continue
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

        epoch_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    print(f'Finished training [{(time.time() - start) / 60:.2f}min]')
    return model
