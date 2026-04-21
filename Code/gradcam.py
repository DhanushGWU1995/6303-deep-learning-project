"""
gradcam.py
----------
Core Grad-CAM implementation. Imported by app.py and 07_gradcam.py.
Supports all four model architectures used in this project, plus the
mouth-attention variants (vgg16_mouth, resnet50_mouth) trained by
09_finetune_mouth_attention.py.
"""

import importlib.util
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models

from config import DEVICE, NUM_CLASSES, DROPOUT_RATE
from train_custom_cnn import CustomCNN

# Lazily import mouth-attention architectures from the fine-tune script
def _load_mouth_attention_classes():
    spec = importlib.util.spec_from_file_location(
        "mouth_attention",
        Path(__file__).parent / "09_finetune_mouth_attention.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.VGG16WithAttention, mod.ResNet50WithAttention

_vgg16_attn_cls = None
_resnet50_attn_cls = None

def _get_vgg16_attn():
    global _vgg16_attn_cls, _resnet50_attn_cls
    if _vgg16_attn_cls is None:
        _vgg16_attn_cls, _resnet50_attn_cls = _load_mouth_attention_classes()
    return _vgg16_attn_cls

def _get_resnet50_attn():
    global _vgg16_attn_cls, _resnet50_attn_cls
    if _resnet50_attn_cls is None:
        _vgg16_attn_cls, _resnet50_attn_cls = _load_mouth_attention_classes()
    return _resnet50_attn_cls


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
    """Return the last convolutional layer for each supported architecture.

    For mouth-attention variants (vgg16_mouth, resnet50_mouth) the CBAM
    spatial-attention conv is hooked — this ensures Grad-CAM activations
    reflect the learned mouth-region weighting.
    """
    if model_name == "custom_cnn":
        return model.features[-1].block[-3]
    elif model_name == "vgg16":
        return model.features[-1]
    elif model_name == "resnet50":
        return model.layer4[-1]
    elif model_name == "efficientnet":
        return model.features[-1]
    elif model_name == "vgg16_mouth":
        # Hook the CBAM spatial-attention conv (after backbone features)
        return model.cbam.spatial_attn.conv
    elif model_name == "resnet50_mouth":
        return model.cbam.spatial_attn.conv
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
    elif model_name == "vgg16_mouth":
        model = _get_vgg16_attn()(NUM_CLASSES)
    elif model_name == "resnet50_mouth":
        model = _get_resnet50_attn()(NUM_CLASSES)
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


# ═══════════════════════════════════════════════════════════════════════════
# Dual-Input Grad-CAM
# ═══════════════════════════════════════════════════════════════════════════

class DualInputGradCAM:
    """
    Grad-CAM for the DualInputPainNet model.

    Generates TWO heat maps — one for the full-face encoder branch and one
    for the mouth-crop encoder branch — so you can compare what each CNN
    attends to.

    Usage::

        gcam = DualInputGradCAM(model)
        full_cam, mouth_cam, pred_idx, conf = gcam.generate(
            full_tensor, mouth_tensor)
        overlay_full  = overlay_heatmap(full_face_rgb,  full_cam)
        overlay_mouth = overlay_heatmap(mouth_crop_rgb, mouth_cam)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        # Last conv block of ResNet-50 (full-face branch)
        # full_face_encoder[-1] = layer4 of ResNet-50 = Sequential(Bottleneck,…)
        self._full_layer   = model.full_face_encoder[-2]   # layer4
        # Last conv block of ResNet-18 (mouth branch)
        self._mouth_layer  = model.mouth_encoder[-2]       # layer4

        self._full_grads  = self._full_acts  = None
        self._mouth_grads = self._mouth_acts = None
        self._register_hooks()

    def _register_hooks(self):
        def _fwd(store):
            def hook(_, __, out): store[0] = out.detach()
            return hook

        def _bwd(store):
            def hook(_, __, grad_out): store[0] = grad_out[0].detach()
            return hook

        self._full_acts_store  = [None]
        self._full_grads_store = [None]
        self._mouth_acts_store  = [None]
        self._mouth_grads_store = [None]

        self._full_layer.register_forward_hook(_fwd(self._full_acts_store))
        self._full_layer.register_full_backward_hook(_bwd(self._full_grads_store))
        self._mouth_layer.register_forward_hook(_fwd(self._mouth_acts_store))
        self._mouth_layer.register_full_backward_hook(_bwd(self._mouth_grads_store))

    @staticmethod
    def _cam_from_stores(acts_store, grads_store) -> np.ndarray:
        acts  = acts_store[0]
        grads = grads_store[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def generate(self, full_tensor: torch.Tensor,
                 mouth_tensor: torch.Tensor,
                 class_idx: int | None = None):
        """
        Returns:
            full_cam   : (H, W) numpy float32 — full-face branch heat map
            mouth_cam  : (H, W) numpy float32 — mouth-branch heat map
            class_idx  : predicted class index
            confidence : softmax confidence for predicted class
        """
        self.model.eval()
        full  = full_tensor.unsqueeze(0).to(DEVICE)
        mouth = mouth_tensor.unsqueeze(0).to(DEVICE)
        full.requires_grad  = True
        mouth.requires_grad = True

        output = self.model(full, mouth)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot)

        full_cam  = self._cam_from_stores(
            self._full_acts_store,  self._full_grads_store)
        mouth_cam = self._cam_from_stores(
            self._mouth_acts_store, self._mouth_grads_store)

        conf = output.softmax(dim=1)[0, class_idx].item()
        return full_cam, mouth_cam, class_idx, conf


def load_dual_input_model(model_path: str) -> nn.Module:
    """Load the DualInputPainNet from a checkpoint."""
    spec = importlib.util.spec_from_file_location(
        "dual_input",
        Path(__file__).parent / "10_train_dual_input.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.DualInputPainNet(NUM_CLASSES, freeze_at_init=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model.to(DEVICE)
