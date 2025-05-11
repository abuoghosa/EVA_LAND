# -----------------------------
# 📦 IMPORT LIBRARIES
# -----------------------------
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# -----------------------------
# ⚙️ CONFIGURATION
# -----------------------------
DATA_DIR = 'eva_land_phase1/data'
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# 🔄 DATA TRANSFORMS & LOADERS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.ImageFolder(
    root=os.path.join(DATA_DIR, 'train'),
    transform=transform
)

val_dataset = torchvision.datasets.ImageFolder(
    root=os.path.join(DATA_DIR, 'val'),
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# 🧠 MODEL SETUP (ResNet)
# -----------------------------
model = resnet18(weights='IMAGENET1K_V1')  # pretrained on ImageNet
model.fc = nn.Linear(model.fc.in_features, 2)  # safe vs unsafe
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------------
# 🏋️‍♂️ TRAINING LOOP
# -----------------------------
def train():
    print(f"🔧 Training on {DEVICE}")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"📈 Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%")

    # Save model weights
    torch.save(model.state_dict(), 'eva_land_phase1/model/cnn_runway_model.pth')
    print("✅ Model training complete. Saved to cnn_runway_model.pth")

# -----------------------------
# 🚀 RUN TRAINING
# -----------------------------
if __name__ == "__main__":
    train()
