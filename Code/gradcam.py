"""
gradcam.py
----------
Core Grad-CAM implementation. Imported by app.py and 07_gradcam.py.
Supports all four model architectures used in this project.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models

from config import DEVICE, NUM_CLASSES, DROPOUT_RATE
from train_custom_cnn import CustomCNN


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, output.softmax(dim=1)[0, class_idx].item()


def get_target_layer(model_name: str, model: nn.Module) -> nn.Module:
    """Return the last convolutional layer for each supported architecture."""
    if model_name == "custom_cnn":
        return model.features[-1].block[-3]
    elif model_name == "vgg16":
        return model.features[-1]
    elif model_name == "resnet50":
        return model.layer4[-1]
    elif model_name == "efficientnet":
        return model.features[-1]
    raise ValueError(f"Unknown model name for Grad-CAM: {model_name}")


def load_model_for_gradcam(model_name: str, model_path: str) -> nn.Module:
    """Load a trained model from disk ready for Grad-CAM inference."""
    if model_name == "custom_cnn":
        model = CustomCNN(NUM_CLASSES)
    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Sequential(
            nn.Dropout(DROPOUT_RATE), nn.Linear(4096, 256),
            nn.ReLU(True), nn.Dropout(DROPOUT_RATE), nn.Linear(256, NUM_CLASSES)
        )
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(DROPOUT_RATE), nn.Linear(2048, 512),
            nn.ReLU(True), nn.Dropout(DROPOUT_RATE), nn.Linear(512, NUM_CLASSES)
        )
    elif model_name == "efficientnet":
        model = models.efficientnet_b3(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE), nn.Linear(1536, 512),
            nn.SiLU(True), nn.Dropout(DROPOUT_RATE), nn.Linear(512, NUM_CLASSES)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model.to(DEVICE)


def overlay_heatmap(original_img: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Blend Grad-CAM heatmap onto the original RGB image."""
    h, w = original_img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * original_img + 0.5 * heatmap_rgb).astype(np.uint8)
    return overlay
