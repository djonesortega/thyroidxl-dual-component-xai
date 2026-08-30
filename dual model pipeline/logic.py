from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = 512
DEFAULT_DROP_RATE = 0.20

FALLBACK_XAI_LAYERS = {
    "Grad-CAM": "backbone.blocks.6.0.conv",
    "Grad-CAM++": "backbone.blocks.5.0.conv_pw",
    "Layer-CAM": "backbone.blocks.6.0.conv",
}

def project_root_from_script(script_file: str | Path) -> Path:
    return Path(script_file).resolve().parent

def _unique_existing(paths):
    output = []
    seen = set()
    for path in paths:
        path = Path(path).resolve()
        if not path.is_file():
            continue
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            output.append(path)
    return output

def find_classifier_checkpoint(project_root: Path) -> Path:
    project_root = Path(project_root)
    models_root = project_root / "models"

    preferred = _unique_existing(
        list(models_root.rglob("*officialtrain9541*final*.pt"))
        + list(models_root.rglob("*mobilenetv3*thyroidxl*best*.pt"))
        + list(models_root.rglob("*mobilenet*thyroidxl*.pt"))
        + list(models_root.rglob("*mobilenet*.pt"))
    )

    valid = []
    for path in preferred:
        try:
            checkpoint = _torch_load_checkpoint(path)
        except Exception:
            continue

        if not isinstance(checkpoint, dict):
            continue

        if str(checkpoint.get("dataset", "")).strip().lower() != "thyroidxl":
            continue

        if "state_dict" not in checkpoint:
            continue

        model_name = str(checkpoint.get("model_name", "")).lower()
        variant = str(checkpoint.get("model_variant", "")).lower()
        if "mobilenet" in model_name or "modelb_v2" in variant:
            valid.append(path)

    if not valid:
        raise FileNotFoundError(
            "Could not find the ThyroidXL MobileNet checkpoint under:\n"
            f"  {models_root}\n\n"
            "Place the saved MobileNetV3 ThyroidXL .pt file anywhere under "
            "the models folder."
        )

    if len(valid) == 1:
        return valid[0]

    ranked = sorted(
        valid,
        key=lambda p: (
            "officialtrain9541" not in p.name.lower(),
            "final" not in p.name.lower(),
            "best" not in p.name.lower(),
            "modelb" not in p.name.lower(),
            len(str(p)),
            str(p).lower(),
        ),
    )
    best = ranked[0]

    best_score = (
        "officialtrain9541" in best.name.lower(),
        "final" in best.name.lower(),
        "best" in best.name.lower(),
        "modelb" in best.name.lower(),
    )
    tied = [
        p for p in ranked
        if (
            "officialtrain9541" in p.name.lower(),
            "final" in p.name.lower(),
            "best" in p.name.lower(),
            "modelb" in p.name.lower(),
        ) == best_score
    ]
    if len(tied) > 1:
        raise RuntimeError(
            "Multiple ThyroidXL MobileNet checkpoints were found. "
            "Set CLASSIFIER_CHECKPOINT in dual_model_pipeline.py.\n\nFound:\n  "
            + "\n  ".join(str(p) for p in valid)
        )

    return best

def find_yolo_checkpoint(project_root: Path) -> Path:
    project_root = Path(project_root)
    models_root = project_root / "models"

    candidates = _unique_existing(
        list((models_root / "YOLOv8sSeg").glob("*selected*maskmap*.pt"))
        + list((models_root / "YOLOv8sSeg").glob("*selected*.pt"))
        + list((models_root / "YOLOv8sSeg").glob("*.pt"))
        + list(models_root.rglob("*yolo*.pt"))
    )

    if not candidates:
        raise FileNotFoundError(
            "Could not find the ThyroidXL YOLOv8s-seg checkpoint under:\n"
            f"  {models_root}\n\n"
            "Expected under models/YOLOv8sSeg/."
        )

    preferred = [
        p for p in candidates
        if "selected" in p.name.lower()
        and (
            "maskmap" in p.name.lower()
            or "mask" in p.name.lower()
        )
    ]
    if len(preferred) == 1:
        return preferred[0]

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        "Multiple YOLO checkpoints were found. Set YOLO_CHECKPOINT in "
        "dual_model_pipeline.py.\n\nFound:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )

def find_json_by_status(
    project_root: Path,
    filename: str,
    expected_status: str,
) -> Path | None:
    matches = []
    for path in (Path(project_root) / "results").rglob(filename):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("status") == expected_status:
            matches.append(path.resolve())

    if not matches:
        return None

    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]

def load_yolo_evaluation_freeze(project_root: Path):
    path = find_json_by_status(
        project_root,
        "yolo_evaluation_freeze.json",
        "THYROIDXL_YOLO_EVALUATION_FROZEN",
    )

    if path is None:
        return None, None

    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, path

def load_xai_freeze(
    project_root: Path,
    model_result_dir: str = "MobileNet",
):
    accepted_statuses = {
        "THYROIDXL_XAI_TURBO_METHOD_SPECIFIC_FROZEN",
        "THYROIDXL_XAI_PROTOCOL_B_METHOD_SPECIFIC_FROZEN",
    }

    base = Path(project_root) / "results" / str(model_result_dir)

    if not base.is_dir():
        return None, None

    matches = []

    for path in base.rglob("protocol_B_freeze.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if payload.get("status") not in accepted_statuses:
            continue

        model_name = str(payload.get("model", "")).strip().lower()

        if (
            model_name
            and str(model_result_dir).lower() == "mobilenet"
            and "mobile" not in model_name
        ):
            continue

        matches.append((path.resolve(), payload))

    if not matches:
        return None, None

    matches.sort(
        key=lambda item: item[0].stat().st_mtime,
        reverse=True,
    )

    path, payload = matches[0]
    return payload, path

def resize_pad(
    image: np.ndarray,
    size: int,
    interpolation_up=cv2.INTER_LINEAR,
    interpolation_down=cv2.INTER_AREA,
):
    if image is None:
        raise ValueError("Image could not be read.")

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape: {image.shape}")

    scale = min(float(size) / float(w), float(size) / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interpolation = (
        interpolation_down if scale < 1.0 else interpolation_up
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=interpolation,
    )

    if image.ndim == 3:
        canvas = np.zeros(
            (size, size, image.shape[2]),
            dtype=resized.dtype,
        )
    else:
        canvas = np.zeros((size, size), dtype=resized.dtype)

    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas

def padding_mask(mask: np.ndarray, size: int):
    return resize_pad(
        mask,
        size,
        interpolation_up=cv2.INTER_NEAREST,
        interpolation_down=cv2.INTER_NEAREST,
    )

def resize_no_pad(
    image: np.ndarray,
    max_size: int,
    interpolation_up=cv2.INTER_LINEAR,
    interpolation_down=cv2.INTER_AREA,
):
    if image is None:
        raise ValueError("Image could not be read.")

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape: {image.shape}")

    scale = float(max_size) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interpolation = (
        interpolation_down if scale < 1.0 else interpolation_up
    )

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=interpolation,
    )

def unpad_square_to_shape(
    padded_map: np.ndarray,
    target_height: int,
    target_width: int,
    interpolation=cv2.INTER_LINEAR,
):
    padded_map = np.asarray(padded_map)

    if padded_map.ndim < 2:
        raise ValueError(
            f"Expected at least 2-D map, got shape {padded_map.shape}"
        )

    square_h, square_w = padded_map.shape[:2]
    if square_h != square_w:
        raise ValueError(
            f"Expected square padded map, got {padded_map.shape[:2]}"
        )

    target_height = int(target_height)
    target_width = int(target_width)

    square_size = int(square_h)

    scale = min(
        float(square_size) / float(target_width),
        float(square_size) / float(target_height),
    )

    content_width = max(
        1,
        min(square_size, int(round(target_width * scale))),
    )
    content_height = max(
        1,
        min(square_size, int(round(target_height * scale))),
    )

    top = (square_size - content_height) // 2
    left = (square_size - content_width) // 2

    cropped = padded_map[
        top:top + content_height,
        left:left + content_width,
    ]

    return cv2.resize(
        cropped,
        (target_width, target_height),
        interpolation=interpolation,
    )

def prepare_display_and_tensor(
    image_path: str | Path,
    input_size: int,
    display_size: int = 640,
):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    display_rgb = resize_no_pad(
        rgb,
        int(display_size),
    )

    model_rgb = resize_pad(
        rgb,
        int(input_size),
    ).astype(np.float32) / 255.0

    mean = np.asarray(IMAGE_MEAN, dtype=np.float32)
    std = np.asarray(IMAGE_STD, dtype=np.float32)

    model_rgb = (model_rgb - mean) / std

    tensor = (
        torch.from_numpy(model_rgb)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(DEVICE)
    )

    return display_rgb, tensor

class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class SkipFusionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.refine = ConvBNAct(
            in_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([x, skip], dim=1)
        return self.refine(x)

class UpsampleRefineBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.refine = ConvBNAct(in_channels, out_channels)

    def forward(self, x, target_size):
        x = F.interpolate(
            x,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.refine(x)

class MobileNetV3MultiscaleDiceBCE(nn.Module):
    def __init__(
        self,
        model_name: str,
        image_size: int = DEFAULT_IMAGE_SIZE,
        drop_rate: float = DEFAULT_DROP_RATE,
    ):
        super().__init__()

        self.image_size = int(image_size)
        self.model_name = str(model_name)
        self.drop_rate = float(drop_rate)

        self.backbone = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=1,
            drop_rate=self.drop_rate,
        )

        was_training = self.backbone.training
        self.backbone.eval()

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                3,
                self.image_size,
                self.image_size,
            )
            final_feature, skips = self._encode(dummy)

        if was_training:
            self.backbone.train()

        final_channels = int(final_feature.shape[1])
        skip32_channels = int(skips[32].shape[1])
        skip64_channels = int(skips[64].shape[1])
        skip128_channels = int(skips[128].shape[1])

        expected_final = (
            self.image_size // 32,
            self.image_size // 32,
        )
        if tuple(final_feature.shape[-2:]) != expected_final:
            raise RuntimeError(
                "Unexpected MobileNetV3 final feature resolution: "
                f"{tuple(final_feature.shape)}"
            )

        self.decoder_bottleneck = ConvBNAct(final_channels, 128)
        self.decoder_skip32 = SkipFusionBlock(
            128,
            skip32_channels,
            96,
        )
        self.decoder_skip64 = SkipFusionBlock(
            96,
            skip64_channels,
            64,
        )
        self.decoder_skip128 = SkipFusionBlock(
            64,
            skip128_channels,
            32,
        )
        self.decoder_up256 = UpsampleRefineBlock(32, 16)
        self.decoder_up512 = UpsampleRefineBlock(16, 8)
        self.segmentation_output = nn.Conv2d(
            8,
            1,
            kernel_size=1,
        )

        self.inferred_feature_channels = {
            "final": final_channels,
            "skip32": skip32_channels,
            "skip64": skip64_channels,
            "skip128": skip128_channels,
        }

    def _encode(self, image):
        x = self.backbone.conv_stem(image)
        x = self.backbone.bn1(x)

        stage_outputs = []
        for block in self.backbone.blocks:
            x = block(x)
            stage_outputs.append(x)

        skips = {}
        for target in (32, 64, 128):
            candidates = [
                feature
                for feature in stage_outputs
                if tuple(feature.shape[-2:]) == (target, target)
            ]

            if not candidates:
                available = sorted(
                    {
                        tuple(feature.shape[-2:])
                        for feature in stage_outputs
                    }
                )
                raise RuntimeError(
                    f"No MobileNetV3 skip feature found at {target}x{target}. "
                    f"Available stage resolutions: {available}"
                )

            skips[target] = candidates[-1]

        return x, skips

    def forward(self, image):
        final_feature, skips = self._encode(image)

        classification_logits = (
            self.backbone.forward_head(final_feature)
            .flatten()
        )

        x = self.decoder_bottleneck(final_feature)
        x = self.decoder_skip32(x, skips[32])
        x = self.decoder_skip64(x, skips[64])
        x = self.decoder_skip128(x, skips[128])

        x = self.decoder_up256(
            x,
            (self.image_size // 2, self.image_size // 2),
        )
        x = self.decoder_up512(
            x,
            (self.image_size, self.image_size),
        )

        segmentation_logits = self.segmentation_output(x)

        if segmentation_logits.shape[-2:] != image.shape[-2:]:
            segmentation_logits = F.interpolate(
                segmentation_logits,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return classification_logits, segmentation_logits

def _torch_load_checkpoint(path: Path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:

        return torch.load(
            path,
            map_location="cpu",
        )

def load_thyroidxl_classifier(
    checkpoint_path: str | Path,
):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {checkpoint_path}"
        )

    checkpoint = _torch_load_checkpoint(checkpoint_path)

    required = {
        "dataset",
        "model_variant",
        "model_name",
        "image_size",
        "drop_rate",
        "state_dict",
        "fold_index",
        "best_validation_patient_auc",
        "validation_patient_threshold",
    }

    missing = sorted(required - set(checkpoint))
    if missing:
        raise KeyError(
            f"Classifier checkpoint is missing required keys: {missing}"
        )

    if str(checkpoint["dataset"]).strip().lower() != "thyroidxl":
        raise RuntimeError(
            f"Expected a ThyroidXL checkpoint, got {checkpoint['dataset']!r}."
        )

    model = MobileNetV3MultiscaleDiceBCE(
        model_name=str(checkpoint["model_name"]),
        image_size=int(checkpoint["image_size"]),
        drop_rate=float(checkpoint.get("drop_rate", DEFAULT_DROP_RATE)),
    )

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    model = model.to(DEVICE).eval()

    return model, checkpoint

class ClassificationOnlyWrapper(nn.Module):
    def __init__(self, trained_model: MobileNetV3MultiscaleDiceBCE):
        super().__init__()
        self.trained_model = trained_model
        self.backbone = trained_model.backbone

    def forward(self, image):

        if hasattr(self.trained_model, "classify"):
            logits = self.trained_model.classify(image)
        else:
            output = self.trained_model(image)
            logits = output[0] if isinstance(output, (tuple, list)) else output
        return logits.reshape(-1, 1)

class PredictedBinaryClassTarget:
    def __init__(self, predicted_class: int):
        self.sign = 1.0 if int(predicted_class) == 1 else -1.0

    def __call__(self, model_output):
        return self.sign * model_output.reshape(-1)[0]

def resolve_named_module(
    model: nn.Module,
    module_name: str,
):
    module_name = str(module_name)

    prefix = "backbone."
    if module_name.startswith(prefix):
        relative = module_name[len(prefix):]
        backbone_modules = dict(model.backbone.named_modules())
        if relative in backbone_modules:
            return backbone_modules[relative]

    modules = dict(model.named_modules(remove_duplicate=False))
    if module_name in modules:
        return modules[module_name]

    likely = sorted(
        {
            f"backbone.{name}"
            for name in dict(model.backbone.named_modules())
            if ("blocks" in name or "conv" in name)
        }
    )

    raise KeyError(
        f"XAI target layer {module_name!r} was not found.\n"
        f"Available candidate modules include:\n  "
        + "\n  ".join(likely[-80:])
    )

def resolve_xai_layers(
    project_root: Path,
    wrapper: ClassificationOnlyWrapper,
):
    freeze, freeze_path = load_xai_freeze(
        project_root,
        model_result_dir="MobileNet",
    )

    method_order = (
        "Grad-CAM",
        "Grad-CAM++",
        "Layer-CAM",
    )

    if freeze is not None:
        selected = freeze.get("selected_per_method", {})

        if not set(method_order).issubset(selected):
            raise RuntimeError(
                "XAI freeze exists but does not contain all three required "
                "CAM methods."
            )

        configs = {}

        for method in method_order:
            frozen = selected[method]

            layer_names = frozen.get("target_layers")

            if not layer_names:
                single = frozen.get("target_layer")
                if single is None:
                    raise RuntimeError(
                        f"Frozen {method} configuration has no target layer(s)."
                    )
                layer_names = [str(single)]

            layer_names = [str(name) for name in layer_names]

            layers = [
                resolve_named_module(wrapper, name)
                for name in layer_names
            ]

            resolutions = frozen.get("target_resolutions")

            if resolutions is None:
                single_resolution = frozen.get("target_resolution")
                resolutions = (
                    [single_resolution]
                    if single_resolution is not None
                    else [None] * len(layer_names)
                )

            configs[method] = {
                "layer_names": layer_names,
                "layers": layers,
                "layer_name": " + ".join(layer_names),
                "layer": layers[0],
                "resolutions": list(resolutions),
                "resolution": (
                    resolutions[0]
                    if len(resolutions) == 1
                    else list(resolutions)
                ),
                "aug_smooth": bool(frozen.get("aug_smooth", False)),
                "eigen_smooth": bool(frozen.get("eigen_smooth", False)),
                "smoothing_name": str(
                    frozen.get("smoothing_name", "none")
                ),
                "config_id": frozen.get("config_id"),
                "source": (
                    "frozen method-specific turbo XAI"
                    if freeze.get("status")
                    == "THYROIDXL_XAI_TURBO_METHOD_SPECIFIC_FROZEN"
                    else "frozen method-specific Protocol B"
                ),
            }

        return configs, freeze_path

    configs = {}

    for method, layer_name in FALLBACK_XAI_LAYERS.items():
        layer = resolve_named_module(wrapper, layer_name)

        configs[method] = {
            "layer_names": [layer_name],
            "layers": [layer],
            "layer_name": layer_name,
            "layer": layer,
            "resolutions": [None],
            "resolution": None,
            "aug_smooth": False,
            "eigen_smooth": False,
            "smoothing_name": "none",
            "config_id": None,
            "source": "development fallback",
        }

    return configs, None

def _cam_from_activation_gradient(
    method_name: str,
    activation: np.ndarray,
    gradient: np.ndarray,
):
    activation = np.asarray(activation, dtype=np.float32)
    gradient = np.asarray(gradient, dtype=np.float32)

    if method_name == "Grad-CAM":
        weights = gradient.mean(axis=(1, 2))
        cam = np.sum(
            activation * weights[:, None, None],
            axis=0,
        )

    elif method_name == "Grad-CAM++":
        g2 = gradient ** 2
        g3 = g2 * gradient

        sum_activations = activation.sum(
            axis=(1, 2),
            keepdims=True,
        )

        denominator = 2.0 * g2 + sum_activations * g3

        denominator = np.where(
            np.abs(denominator) > 1e-7,
            denominator,
            1.0,
        )

        aij = g2 / denominator
        aij = np.where(gradient != 0.0, aij, 0.0)

        weights = (
            np.maximum(gradient, 0.0) * aij
        ).sum(axis=(1, 2))

        cam = np.sum(
            activation * weights[:, None, None],
            axis=0,
        )

    elif method_name == "Layer-CAM":
        cam = np.sum(
            activation * np.maximum(gradient, 0.0),
            axis=0,
        )

    else:
        raise KeyError(
            f"Unsupported CAM method: {method_name}"
        )

    return np.maximum(cam, 0.0).astype(np.float32)

def generate_frozen_cam_maps(
    wrapper: ClassificationOnlyWrapper,
    image_tensor: torch.Tensor,
    predicted_class: int,
    xai_configs: dict,
):
    method_order = (
        "Grad-CAM",
        "Grad-CAM++",
        "Layer-CAM",
    )

    missing = [
        method for method in method_order
        if method not in xai_configs
    ]

    if missing:
        raise RuntimeError(
            f"Missing frozen XAI configurations: {missing}"
        )

    unique_layers = {}

    for config in xai_configs.values():
        for name, layer in zip(
            config["layer_names"],
            config["layers"],
        ):
            unique_layers[str(name)] = layer

    activations = {}
    gradients = {}
    handles = []

    def hook_factory(layer_name):
        def hook(_module, _inputs, output):
            activations[layer_name] = output

            if output.requires_grad:
                output.register_hook(
                    lambda grad, name=layer_name:
                    gradients.__setitem__(name, grad)
                )

        return hook

    for layer_name, layer in unique_layers.items():
        handles.append(
            layer.register_forward_hook(
                hook_factory(layer_name)
            )
        )

    try:
        wrapper.zero_grad(set_to_none=True)

        logits = wrapper(image_tensor).reshape(-1)

        if logits.numel() != 1:
            raise RuntimeError(
                "Dual-model visualiser expects exactly one image per CAM run."
            )

        sign = 1.0 if int(predicted_class) == 1 else -1.0
        (logits[0] * float(sign)).backward()

    finally:
        for handle in handles:
            handle.remove()

    model_h = int(image_tensor.shape[-2])
    model_w = int(image_tensor.shape[-1])

    cam_maps = {}

    for method in method_order:
        config = xai_configs[method]
        layer_maps = []

        for layer_name in config["layer_names"]:
            if (
                layer_name not in activations
                or layer_name not in gradients
            ):
                raise RuntimeError(
                    f"CAM hook failed for frozen layer: {layer_name}"
                )

            activation = (
                activations[layer_name][0]
                .detach()
                .cpu()
                .numpy()
            )
            gradient = (
                gradients[layer_name][0]
                .detach()
                .cpu()
                .numpy()
            )

            raw = _cam_from_activation_gradient(
                method,
                activation,
                gradient,
            )

            resized = cv2.resize(
                raw,
                (model_w, model_h),
                interpolation=cv2.INTER_LINEAR,
            )

            layer_maps.append(norm01(resized))

        if len(layer_maps) == 1:
            combined = layer_maps[0]
        else:
            combined = norm01(
                np.mean(
                    np.stack(layer_maps, axis=0),
                    axis=0,
                )
            )

        cam_maps[method] = norm01(combined)

    return cam_maps

def norm01(values: np.ndarray):
    values = np.asarray(values, dtype=np.float32)
    minimum = float(values.min())
    maximum = float(values.max())

    if maximum <= minimum:
        return np.zeros_like(values, dtype=np.float32)

    return (values - minimum) / (maximum - minimum)

def overlay_cam_on_image(
    image_rgb: np.ndarray,
    cam_01: np.ndarray,
    alpha: float = 0.45,
):
    cam_01 = np.clip(
        np.asarray(cam_01, dtype=np.float32),
        0.0,
        1.0,
    )

    heat = cv2.applyColorMap(
        np.uint8(cam_01 * 255.0),
        cv2.COLORMAP_JET,
    )
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(
        image_rgb,
        1.0 - alpha,
        heat,
        alpha,
        0.0,
    )

def overlay_mask_on_image(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.42,
):
    image = image_rgb.copy()
    mask_bool = np.asarray(mask).astype(bool)

    if not mask_bool.any():
        return image

    overlay = image.astype(np.float32)
    overlay[mask_bool] = (
        (1.0 - alpha) * overlay[mask_bool]
        + alpha * np.asarray([255.0, 255.0, 255.0], dtype=np.float32)
    )

    output = np.clip(overlay, 0, 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_bool.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.drawContours(
        bgr,
        contours,
        -1,
        (255, 255, 255),
        2,
    )

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def top_fraction_binary(
    activation_map: np.ndarray,
    fraction: float = 0.15,
):
    values = norm01(activation_map)
    flat = values.reshape(-1)

    if flat.size == 0:
        return np.zeros_like(values, dtype=bool)

    k = max(1, int(np.ceil(float(fraction) * flat.size)))
    kth = max(0, flat.size - k)
    threshold = np.partition(flat, kth)[kth]

    return values >= threshold

def binary_iou(a, b):
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)

    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")

    intersection = np.logical_and(a, b).sum()
    return float(intersection / union)

def cam_yolo_top15_iou(cam_map, yolo_mask):
    if not np.asarray(yolo_mask).astype(bool).any():
        return float("nan")
    return binary_iou(
        top_fraction_binary(cam_map, fraction=0.15),
        yolo_mask,
    )
