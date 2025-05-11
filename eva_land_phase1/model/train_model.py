# -----------------------------
#IMPORT LIBRARIES
# -----------------------------
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR = 'eva_land_phase1/data'
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# DATA TRANSFORMS AND LOADERS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
val_dataset = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -----------------------------
# MODEL SETUP
# -----------------------------
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification: runway / non-runway
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------------
# TRAINING LOOP
# -----------------------------
def train():
    print(f"Training on {DEVICE}")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), 'cnn_runway_model.pth')
    print("Model training complete and saved as cnn_runway_model.pth")

if __name__ == "__main__":
    train()

