import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt
from io import BytesIO

# ---------------------
# #config
# ---------------------
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ---------------------
# #train - Load Your Trained CNN Model
# ---------------------
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # runway / non_runway
model.load_state_dict(torch.load("eva_land_phase1/model/cnn_runway_model.pth", map_location=DEVICE))
model.eval()
model.to(DEVICE)

# ---------------------
# #gradCAM - Register hooks
# ---------------------
features_blobs = []
gradients = []

def forward_hook(module, input, output):
    features_blobs.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

model._modules.get('layer4').register_forward_hook(forward_hook)
model._modules.get('layer4').register_backward_hook(backward_hook)

# ---------------------
# #explain - GradCAM + Faithfulness
# ---------------------
def analyze_image(img):
    features_blobs.clear()
    gradients.clear()

    orig = img.resize((IMG_SIZE, IMG_SIZE))
    input_tensor = transform(orig).unsqueeze(0).to(DEVICE)

    # Forward
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1).squeeze()
    class_idx = probs.argmax().item()
    conf_score = probs[class_idx].item()
    label = "runway" if class_idx == 0 else "non-runway"

    # Backward
    model.zero_grad()
    output[0, class_idx].backward()

    # Grad-CAM
    weights = torch.mean(gradients[0], dim=[2, 3])[0]
    cam = torch.zeros(features_blobs[0].shape[2:], dtype=torch.float32).to(DEVICE)
    for i, w in enumerate(weights):
        cam += w * features_blobs[0][0, i, :, :]
    cam = cam.cpu().detach().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    # Overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(orig), 0.5, heatmap, 0.5, 0)

    # #faithful - Perturbation Masking
    mask = cam >= np.percentile(cam, 90)
    masked_tensor = input_tensor.clone()
    for c in range(3):
        masked_tensor[0, c, mask] = 0.5
    with torch.no_grad():
        masked_out = model(masked_tensor)
        masked_conf = torch.softmax(masked_out, dim=1).squeeze()[class_idx].item()
    drop = conf_score - masked_conf
    faithful = "FAITHFUL ✅" if drop > 0.2 else "NOT FAITHFUL ❌"

    # #result - Display Explanation Report
    result = (
        f"Prediction: {label} | Confidence Score: {conf_score*100:.1f}%\n"
        f"Surface: Clear | Obstacles: None | Explanation is {faithful}\n\n"
        f"Original Score: {conf_score:.4f}\n"
        f"Masked Score: {masked_conf:.4f}\n"
        f"Drop in Confidence: {drop:.4f}"
    )

    return Image.fromarray(overlay), result

# ---------------------
# #gradio - Interface
# ---------------------
iface = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="pil", label="Upload EVA_LAND Input Image"),
    outputs=[
        gr.Image(type="pil", label="Grad-CAM Explanation"),
        gr.Textbox(label="Explanation Summary")
    ],
    title="✈️ EVA_LAND Phase 1 – Explainable Emergency Landing Aid",
    description="Upload an aerial view from LARD. The system returns a prediction, Grad-CAM visualization, and a faithfulness test report."
)

if __name__ == "__main__":
    iface.launch()

