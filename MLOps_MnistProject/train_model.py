import os
from datetime import datetime as dt

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from models.model import MyNeuralNet


def train_step(model, optimizer, criterion, images, labels):
    images = images.unsqueeze(1)
    optimizer.zero_grad()
    y_pred = model(images)
    loss = criterion(y_pred, labels)
    loss.backward()
    optimizer.step()
    return loss

def train():
    data_path = "data/processed"
    file_path = os.path.join(data_path)
    train_dataset = torch.load(f"{file_path}/train_dataset.pt")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = MyNeuralNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 40

    train_losses = []
    for epoch in range(epochs):
        train_loss = 0
        for images, labels in train_loader:
            loss = train_step(model, optimizer, criterion, images, labels)
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch: {epoch}, Loss: {avg_train_loss}")

    timestamp = dt.now().strftime("%Y%m%d%H%M%S")

    torch.save(model, f"models/model_mnist{timestamp}.pt")
    torch.save(model, "models/model_mnist_latest.pt")

    # Plot the training loss
    plt.plot(range(epochs), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs. Epoch")
    plt.savefig(f"reports/figures/training_loss_mnist_{timestamp}.png")


if __name__ == "__main__":
    train()
