import torch
import numpy as np
import cv2
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
TEST_DIR = Path("eva_land_phase1/data/test")  # LARD test directory
OUTPUT_DIR = Path("eva_land_phase1/outputs/gradcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Transform and Model Setup
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
model.to(DEVICE)

# ---------------------------
# Hook Setup
# ---------------------------
features_blobs = []
gradients = []

def forward_hook(module, input, output):
    features_blobs.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

finalconv_name = 'layer4'
model._modules.get(finalconv_name).register_forward_hook(forward_hook)
model._modules.get(finalconv_name).register_backward_hook(backward_hook)

# ---------------------------
# Grad-CAM Generator
# ---------------------------
def generate_gradcam(image_tensor):
    global features_blobs, gradients
    features_blobs, gradients = [], []

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    output = model(image_tensor)
    class_idx = output.argmax().item()

    model.zero_grad()
    output[0, class_idx].backward()

    weights = torch.mean(gradients[0], dim=[2, 3])[0]
    cam = torch.zeros(features_blobs[0].shape[2:], dtype=torch.float32).to(DEVICE)

    for i, w in enumerate(weights):
        cam += w * features_blobs[0][0, i, :, :]

    cam = cam.cpu().detach().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    return cam, class_idx

# ---------------------------
# Process Entire Dataset
# ---------------------------
for class_folder in TEST_DIR.iterdir():
    if not class_folder.is_dir():
        continue
    out_class_dir = OUTPUT_DIR / class_folder.name
    out_class_dir.mkdir(parents=True, exist_ok=True)

    for img_path in class_folder.glob("*.jpg"):
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image)
            cam, predicted_class = generate_gradcam(image_tensor)

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            original = np.array(image.resize((224, 224)))
            overlay = cv2.addWeighted(original, 0.5, heatmap, 0.5, 0)

            out_path = out_class_dir / f"{img_path.stem}_gradcam.jpg"
            cv2.imwrite(str(out_path), overlay)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

print("✅ Grad-CAM visualizations saved to:", OUTPUT_DIR)
