import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from data.cifar10_loader import get_cifar10_loaders
from models.simple_cnn import SimpleCNN


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def train_model(
    epochs=10, lr=0.001, device="cuda" if torch.cuda.is_available() else "cpu"
):
    trainloader, valloader, testloader = get_cifar10_loaders()
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_acc = evaluate(model, valloader, device)
        print(
            f"Epoch {epoch+1} | Loss: {running_loss/len(trainloader):.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "experiments/baseline_model.pt")

    print("✅ Best model saved to experiments/baseline_model.pt")

    # Final test evaluation
    model.load_state_dict(torch.load("experiments/baseline_model.pt"))
    test_acc = evaluate(model, testloader, device)
    print(f"🎯 Final Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    train_model()
