import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data.cifar10_loader import get_cifar10_loaders
from models.simple_cnn import SimpleCNN

def train_model(epochs=5, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    trainloader, testloader = get_cifar10_loaders()
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
        print(f"Epoch {epoch+1} loss: {running_loss/len(trainloader):.4f}")

    # Save model
    torch.save(model.state_dict(), "experiments/baseline_model.pt")
    print("✅ Baseline model saved!")

if __name__ == "__main__":
    train_model()