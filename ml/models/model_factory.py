# ml/models/model_factory.py
"""
Build a segmentation model by name.

Supported MODEL_TYPE values
───────────────────────────
  "custom_unet"   UNetSmall trained from scratch (~490K params). No ImageNet norm.
  "smp_resnet18"  smp.Unet, ResNet-18 encoder, ImageNet weights.
  "smp_effnet_b0" smp.Unet, EfficientNet-B0 encoder, ImageNet weights.
"""

import torch
import torch.nn as nn

# ImageNet statistics — applied when encoder was pretrained on ImageNet.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_SMP_MODELS = {"smp_resnet18", "smp_effnet_b0"}

_SMP_ENCODER = {
    "smp_resnet18":   "resnet18",
    "smp_effnet_b0":  "efficientnet-b0",
}


def build_model(model_type: str) -> nn.Module:
    if model_type == "custom_unet":
        from ml.models.unet import UNetSmall
        return UNetSmall()

    if model_type in _SMP_MODELS:
        import segmentation_models_pytorch as smp
        return smp.Unet(
            encoder_name=_SMP_ENCODER[model_type],
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    raise ValueError(
        f"Unknown MODEL_TYPE: {model_type!r}. "
        f"Choose from: custom_unet, smp_resnet18, smp_effnet_b0"
    )


def uses_imagenet_norm(model_type: str) -> bool:
    """True when the encoder was pretrained on ImageNet and needs ImageNet normalization."""
    return model_type in _SMP_MODELS


def select_device(model: nn.Module) -> tuple[nn.Module, str]:
    """
    Move model to MPS if available. If a MPS forward pass fails (rare op
    incompatibility), falls back to CPU with a printed warning.
    """
    if not torch.backends.mps.is_available():
        return model.to("cpu"), "cpu"

    try:
        model = model.to("mps")
        with torch.no_grad():
            model(torch.zeros(1, 3, 64, 64, device="mps"))
        return model, "mps"
    except Exception as exc:
        print(f"[warn] MPS forward failed ({exc}); falling back to CPU.")
        model = model.to("cpu")
        return model, "cpu"
