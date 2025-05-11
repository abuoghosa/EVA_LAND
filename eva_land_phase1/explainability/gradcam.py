import torch
import numpy as np
import cv2
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image

# === Load and Transform Image ===
image = Image.open("test.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
input_tensor = transform(image).unsqueeze(0)

# === Load pretrained ResNet ===
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# === Hook Functions ===
features_blobs = []
gradients = []

def forward_hook(module, input, output):
    features_blobs.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

# Register hooks to the last conv layer
finalconv_name = 'layer4'
model._modules.get(finalconv_name).register_forward_hook(forward_hook)
model._modules.get(finalconv_name).register_backward_hook(backward_hook)

# === Forward Pass ===
output = model(input_tensor)
class_idx = output.argmax().item()

# === Backward Pass ===
model.zero_grad()
output[0, class_idx].backward()

# === Compute Grad-CAM ===
weights = torch.mean(gradients[0], dim=[2, 3])[0]
cam = torch.zeros(features_blobs[0].shape[2:], dtype=torch.float32)

for i, w in enumerate(weights):
    cam += w * features_blobs[0][0, i, :, :]

cam = np.maximum(cam.detach().numpy(), 0)
cam = cv2.resize(cam, (224, 224))
cam -= cam.min()
cam /= cam.max()

# === Overlay on Original Image ===
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
original_img = np.array(image.resize((224, 224)))
overlay = cv2.addWeighted(original_img, 0.5, heatmap, 0.5, 0)

# === Show Result ===
plt.imshow(overlay)
plt.title("Emergency Landing Decision Heatmap (Grad-CAM)")
plt.axis('off')
plt.show()
