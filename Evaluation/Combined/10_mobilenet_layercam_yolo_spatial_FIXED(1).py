from __future__ import annotations

"""
ThyroidXL — MobileNetV3 Layer-CAM vs YOLOv8s-seg spatial agreement
===================================================================

Purpose
-------
Quantify whether the FINAL frozen MobileNetV3 Layer-CAM explanation and the
FINAL YOLOv8s-seg nodule mask localise the same region on the official held-out
ThyroidXL test cohort.

This is SPATIAL agreement, not merely classifier-label agreement.

Locked inputs
-------------
MobileNet Layer-CAM:
    32x32 + 16x16 target layers
    predicted-class logit target at the fixed 0.5 image threshold
    top 15% of the normalised CAM for binary spatial overlap metrics

YOLO:
    final official-train-refit YOLOv8s-seg checkpoint
    SHA256 = a9dcd76aa76fa91ae2b9d61820fdd53a06df7e98f10efb5bb2dea8efdc01e92b
    image size = 640
    retrieval confidence = 0.001
    NMS IoU = 0.70
    max detections = 300
    TTA/augment = False
    spatial mask = highest-confidence retained detection mask

Coordinate system
-----------------
Both Layer-CAM and YOLO masks are mapped to the ORIGINAL ultrasound geometry
before comparison.

Primary agreement outputs
-------------------------
1. Layer-CAM top-15% mask vs YOLO mask IoU
2. Layer-CAM top-15% mask vs YOLO mask Dice
3. Layer-CAM/Yolo any-overlap rate
4. Layer-CAM peak-inside-YOLO-mask rate
5. Fraction of Layer-CAM energy inside the YOLO mask

Context outputs
---------------
- Layer-CAM vs expert nodule mask
- YOLO mask vs expert nodule mask
- overall and diagnosis-stratified patient-balanced summaries
- patient-bootstrap 95% confidence intervals
- qualitative agreement panels

No threshold/layer/model setting is selected from the official test.
"""

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO

try:
    from pytorch_grad_cam import LayerCAM
except ImportError as exc:
    raise ImportError(
        "Missing XAI dependency. Install it with:\n"
        "  pip install grad-cam"
    ) from exc


# =============================================================================
# Frozen protocol
# =============================================================================

SEED = 42

EXPECTED_TEST_IMAGES = 2094
EXPECTED_TEST_PATIENTS = 739

MOBILENET_FRAME_THRESHOLD = 0.50
LAYERCAM_RESOLUTIONS = (32, 16)
LAYERCAM_TOP_FRACTION = 0.15

YOLO_EXPECTED_SHA256 = (
    "a9dcd76aa76fa91ae2b9d61820fdd53"
    "a06df7e98f10efb5bb2dea8efdc01e92b"
)
YOLO_IMAGE_SIZE = 640
YOLO_RETRIEVAL_CONFIDENCE = 0.001
YOLO_NMS_IOU = 0.70
YOLO_MAX_DETECTIONS = 300
YOLO_AUGMENT = False

BOOTSTRAP_SAMPLES = 2000
QUALITATIVE_PATIENTS_PER_CLASS = 6

PROTOCOL_VERSION = "mobilenet_layercam32_16_yolo_topmask_original_geometry_v1"


# =============================================================================
# Project and MobileNet evaluator discovery
# =============================================================================

def find_project_root(script_file: str | Path) -> Path:
    script_path = Path(script_file).resolve()

    for candidate in [script_path.parent, *script_path.parents]:
        if (
            (candidate / "Evaluation").is_dir()
            and (candidate / "Models").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate ThyroidXL project root. Expected a parent containing "
        "Evaluation/ and Models/."
    )


ROOT = find_project_root(__file__)


def load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from: {path}")

    module = importlib.util.module_from_spec(spec)

    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    spec.loader.exec_module(module)
    return module


MOBILENET_EVALUATOR = (
    ROOT
    / "Evaluation"
    / "MobileNet"
    / "mobilenetv3_xai_evaluation.py"
)

if not MOBILENET_EVALUATOR.is_file():
    raise FileNotFoundError(
        "Expected MobileNet XAI evaluator at:\n"
        f"  {MOBILENET_EVALUATOR}"
    )

M = load_module_from_path(
    MOBILENET_EVALUATOR,
    "_thyroidxl_mobilenet_xai_for_agreement",
)


# =============================================================================
# Output
# =============================================================================

OUT = (
    ROOT
    / "results"
    / "Combined"
    / "MobileNet_LayerCAM_YOLO_Agreement"
)
OUT.mkdir(parents=True, exist_ok=True)

QUAL_OUT = OUT / "qualitative"
QUAL_OUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# General helpers
# =============================================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_cam(cam):
    cam = np.asarray(cam, dtype=np.float32)
    low = float(cam.min())
    high = float(cam.max())

    if (
        not np.isfinite(low)
        or not np.isfinite(high)
        or high <= low
    ):
        return np.zeros_like(cam, dtype=np.float32)

    return (
        (cam - low)
        / (high - low)
    ).astype(np.float32)


def top_fraction_mask(cam, fraction=LAYERCAM_TOP_FRACTION):
    flat = np.asarray(cam, dtype=np.float32).reshape(-1)
    active = max(
        1,
        int(math.ceil(flat.size * float(fraction))),
    )
    indices = np.argpartition(
        flat,
        -active,
    )[-active:]

    output = np.zeros(
        flat.size,
        dtype=np.uint8,
    )
    output[indices] = 1

    return output.reshape(cam.shape).astype(bool)


def as_2d_binary_mask(mask, name="mask"):
    """Return a strict 2D boolean mask, tolerating singleton channel axes."""
    array = np.asarray(mask)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise RuntimeError(
            f"{name} must resolve to a 2D mask after squeezing singleton axes; "
            f"got shape {np.asarray(mask).shape} -> {array.shape}."
        )
    return array > 0


def binary_mask_metrics(first, second):
    first = as_2d_binary_mask(first, "first mask")
    second = as_2d_binary_mask(second, "second mask")
    if first.shape != second.shape:
        raise RuntimeError(
            f"Binary-mask geometry mismatch: first={first.shape}, second={second.shape}."
        )

    intersection = float(
        np.logical_and(first, second).sum()
    )
    union = float(
        np.logical_or(first, second).sum()
    )

    first_sum = float(first.sum())
    second_sum = float(second.sum())

    iou = (
        intersection / union
        if union > 0.0
        else 0.0
    )

    dice = (
        2.0 * intersection
        / (first_sum + second_sum)
        if (first_sum + second_sum) > 0.0
        else 0.0
    )

    any_overlap = float(
        intersection > 0.0
    )

    return {
        "iou": float(iou),
        "dice": float(dice),
        "any_overlap": float(any_overlap),
        "intersection_pixels": int(intersection),
        "first_pixels": int(first_sum),
        "second_pixels": int(second_sum),
    }


def layercam_yolo_metrics(cam, yolo_mask):
    cam = normalise_cam(cam)
    cam = np.squeeze(np.asarray(cam, dtype=np.float32))
    if cam.ndim != 2:
        raise RuntimeError(f"Layer-CAM must be 2D; got shape {cam.shape}.")
    yolo_mask = as_2d_binary_mask(yolo_mask, "YOLO mask")
    if cam.shape != yolo_mask.shape:
        raise RuntimeError(
            f"Layer-CAM/YOLO geometry mismatch: CAM={cam.shape}, YOLO={yolo_mask.shape}."
        )
    cam_binary = top_fraction_mask(cam)

    spatial = binary_mask_metrics(
        cam_binary,
        yolo_mask,
    )

    peak = np.unravel_index(
        int(np.argmax(cam)),
        cam.shape,
    )

    peak_inside = (
        float(yolo_mask[peak])
        if yolo_mask.any()
        else 0.0
    )

    nonnegative = np.clip(
        cam,
        0.0,
        None,
    )
    total_energy = float(
        nonnegative.sum()
    )

    energy_inside = (
        float(
            nonnegative[yolo_mask].sum()
            / total_energy
        )
        if total_energy > 0.0
        and yolo_mask.any()
        else 0.0
    )

    inside_mean = (
        float(cam[yolo_mask].mean())
        if yolo_mask.any()
        else 0.0
    )
    outside_mean = (
        float(cam[~yolo_mask].mean())
        if (~yolo_mask).any()
        else 0.0
    )

    return {
        "layercam_yolo_top15_iou": spatial["iou"],
        "layercam_yolo_top15_dice": spatial["dice"],
        "layercam_yolo_any_overlap": spatial["any_overlap"],
        "layercam_peak_inside_yolo": peak_inside,
        "layercam_energy_inside_yolo": energy_inside,
        "layercam_mean_inside_yolo": inside_mean,
        "layercam_mean_outside_yolo": outside_mean,
        "layercam_yolo_inside_outside_ratio": float(
            inside_mean
            / (outside_mean + 1e-8)
        ),
        "layercam_top15_pixels": spatial["first_pixels"],
        "yolo_mask_pixels": spatial["second_pixels"],
        "layercam_yolo_intersection_pixels": spatial["intersection_pixels"],
    }


# =============================================================================
# YOLO checkpoint and inference
# =============================================================================

def find_final_yolo_checkpoint():
    model_root = ROOT / "Models" / "YOLOv8sSeg"

    if not model_root.is_dir():
        raise FileNotFoundError(
            f"Missing YOLO model directory: {model_root}"
        )

    exact = []

    for path in model_root.rglob("*.pt"):
        if not path.is_file():
            continue

        try:
            current = sha256_file(path)
        except Exception:
            continue

        if current == YOLO_EXPECTED_SHA256:
            exact.append(path.resolve())

    if len(exact) != 1:
        raise RuntimeError(
            "Expected exactly one final YOLO checkpoint with SHA256:\n"
            f"  {YOLO_EXPECTED_SHA256}\n"
            f"Found {len(exact)}:\n"
            + "\n".join(str(path) for path in exact)
        )

    return exact[0]


def yolo_top_mask(
    yolo_model,
    image_path: Path,
    original_shape,
    device,
):
    """
    Use the highest-confidence retained YOLO detection, matching the final YOLO
    publication evaluator's segmentation-mask reporting rule.
    """
    result = yolo_model.predict(
        source=str(image_path),
        imgsz=int(YOLO_IMAGE_SIZE),
        conf=float(YOLO_RETRIEVAL_CONFIDENCE),
        iou=float(YOLO_NMS_IOU),
        max_det=int(YOLO_MAX_DETECTIONS),
        device=device,
        augment=bool(YOLO_AUGMENT),
        retina_masks=True,
        verbose=False,
    )[0]

    height, width = (
        int(original_shape[0]),
        int(original_shape[1]),
    )

    empty = np.zeros(
        (height, width),
        dtype=bool,
    )

    boxes = result.boxes

    if (
        boxes is None
        or len(boxes) == 0
    ):
        return {
            "mask": empty,
            "has_detection": False,
            "n_detections": 0,
            "top_confidence": 0.0,
            "top_class_id": None,
            "top_class_name": "",
        }

    confidences = (
        boxes.conf
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )
    classes = (
        boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    top_idx = int(
        np.argmax(confidences)
    )

    top_confidence = float(
        confidences[top_idx]
    )
    top_class_id = int(
        classes[top_idx]
    )

    names = result.names

    if isinstance(names, dict):
        top_class_name = str(
            names.get(
                top_class_id,
                "",
            )
        )
    elif (
        isinstance(names, (list, tuple))
        and 0 <= top_class_id < len(names)
    ):
        top_class_name = str(
            names[top_class_id]
        )
    else:
        top_class_name = ""

    mask = empty

    if (
        result.masks is not None
        and len(result.masks.data) > top_idx
    ):
        raw = (
            result.masks.data[top_idx]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if raw.shape != (height, width):
            raw = cv2.resize(
                raw,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

        mask = raw > 0.5

    return {
        "mask": mask,
        "has_detection": True,
        "n_detections": int(len(boxes)),
        "top_confidence": top_confidence,
        "top_class_id": top_class_id,
        "top_class_name": top_class_name,
    }


# =============================================================================
# Patient-balanced summaries / bootstrap
# =============================================================================

PRIMARY_METRICS = (
    "layercam_yolo_top15_iou",
    "layercam_yolo_top15_dice",
    "layercam_yolo_any_overlap",
    "layercam_peak_inside_yolo",
    "layercam_energy_inside_yolo",
)

CONTEXT_METRICS = (
    "layercam_expert_top15_iou",
    "layercam_expert_strict_pointing_hit",
    "layercam_expert_nodule_energy_fraction",
    "yolo_expert_mask_iou",
    "yolo_expert_mask_dice",
)

SUMMARY_METRICS = PRIMARY_METRICS + CONTEXT_METRICS


def patient_balanced_summary(frame):
    patient = (
        frame.groupby(
            "patient_id"
        )[list(SUMMARY_METRICS)]
        .mean()
    )

    output = {
        "n_images": int(len(frame)),
        "n_patients": int(
            frame["patient_id"].nunique()
        ),
        "yolo_detection_coverage": float(
            frame["yolo_has_detection"].mean()
        ),
    }

    for metric in SUMMARY_METRICS:
        output[
            "patient_balanced_mean_"
            + metric
        ] = float(
            patient[metric].mean()
        )

    return output


def patient_bootstrap_ci(
    frame,
    metrics=PRIMARY_METRICS,
    samples=BOOTSTRAP_SAMPLES,
    seed=SEED,
):
    patient = (
        frame.groupby(
            "patient_id"
        )[list(metrics)]
        .mean()
        .sort_index()
    )

    values = patient.to_numpy(
        dtype=np.float64
    )

    n = values.shape[0]

    rng = np.random.default_rng(
        int(seed)
    )

    boot = np.empty(
        (
            int(samples),
            len(metrics),
        ),
        dtype=np.float64,
    )

    for index in range(
        int(samples)
    ):
        draw = rng.integers(
            0,
            n,
            size=n,
        )

        boot[index] = (
            values[draw]
            .mean(axis=0)
        )

    low = np.quantile(
        boot,
        0.025,
        axis=0,
    )
    high = np.quantile(
        boot,
        0.975,
        axis=0,
    )
    estimate = values.mean(
        axis=0
    )

    return {
        metric: {
            "estimate": float(
                estimate[index]
            ),
            "ci95_low": float(
                low[index]
            ),
            "ci95_high": float(
                high[index]
            ),
        }
        for index, metric
        in enumerate(metrics)
    }


# =============================================================================
# Qualitative cases
# =============================================================================

def choose_qualitative_patients(test_frame):
    patients = (
        test_frame[
            ["patient_id", "label"]
        ]
        .drop_duplicates()
        .copy()
    )

    rng = np.random.default_rng(
        SEED
    )

    selected = set()

    for label in (0, 1):
        ids = (
            patients.loc[
                patients["label"] == label,
                "patient_id",
            ]
            .astype(str)
            .to_numpy()
        )

        rng.shuffle(ids)

        selected.update(
            ids[
                :QUALITATIVE_PATIENTS_PER_CLASS
            ].tolist()
        )

    return selected


def save_qualitative_panel(
    filename,
    patient_id,
    label,
    image_path,
    expert_mask,
    cam,
    yolo_mask,
    row_metrics,
):
    image_bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        return

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    cam = normalise_cam(
        cam
    )
    cam_binary = top_fraction_mask(
        cam
    )

    heat = cv2.applyColorMap(
        np.uint8(
            cam * 255.0
        ),
        cv2.COLORMAP_JET,
    )
    heat = cv2.cvtColor(
        heat,
        cv2.COLOR_BGR2RGB,
    )

    cam_overlay = cv2.addWeighted(
        image_rgb,
        0.55,
        heat,
        0.45,
        0.0,
    )

    combined = image_rgb.copy()

    expert_contours, _ = cv2.findContours(
        (
            np.asarray(expert_mask) > 0
        ).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    yolo_contours, _ = cv2.findContours(
        (
            np.asarray(yolo_mask) > 0
        ).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    combined_bgr = cv2.cvtColor(
        combined,
        cv2.COLOR_RGB2BGR,
    )

    # Expert = white; YOLO = thicker dark boundary.
    cv2.drawContours(
        combined_bgr,
        expert_contours,
        -1,
        (255, 255, 255),
        2,
    )
    cv2.drawContours(
        combined_bgr,
        yolo_contours,
        -1,
        (0, 0, 0),
        3,
    )

    combined = cv2.cvtColor(
        combined_bgr,
        cv2.COLOR_BGR2RGB,
    )

    figure, axes = plt.subplots(
        1,
        5,
        figsize=(18, 4),
    )

    axes[0].imshow(image_rgb)
    axes[0].set_title("Ultrasound")

    axes[1].imshow(expert_mask, cmap="gray")
    axes[1].set_title("Expert mask")

    axes[2].imshow(cam_overlay)
    axes[2].contour(
        cam_binary.astype(np.uint8),
        levels=[0.5],
        linewidths=1,
    )
    axes[2].set_title("MobileNet Layer-CAM")

    axes[3].imshow(yolo_mask, cmap="gray")
    axes[3].set_title("YOLO top mask")

    axes[4].imshow(combined)
    axes[4].imshow(
        np.ma.masked_where(
            ~cam_binary,
            cam_binary,
        ),
        alpha=0.25,
        cmap="autumn",
    )
    axes[4].set_title(
        "Spatial agreement\n"
        f"IoU={row_metrics['layercam_yolo_top15_iou']:.3f}, "
        f"Dice={row_metrics['layercam_yolo_top15_dice']:.3f}"
    )

    for axis in axes:
        axis.axis("off")

    diagnosis = (
        "malignant"
        if int(label) == 1
        else "benign"
    )

    figure.suptitle(
        f"Patient {patient_id} | {diagnosis} | {filename}"
    )

    figure.tight_layout()

    figure.savefig(
        QUAL_OUT
        / f"{patient_id}_{filename}.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("THYROIDXL — MOBILENET LAYER-CAM vs YOLO SPATIAL AGREEMENT")
    print("=" * 80)
    print("Project root:", ROOT)
    print("MobileNet evaluator:", MOBILENET_EVALUATOR)
    print()
    print("Locked MobileNet Layer-CAM: 32x32 + 16x16")
    print("Layer-CAM binary region: top 15% activation")
    print("YOLO mask: highest-confidence retained detection")
    print("Comparison geometry: ORIGINAL ULTRASOUND")
    print("No official-test tuning: YES")
    print()

    # ------------------------------------------------------------------
    # MobileNet
    # ------------------------------------------------------------------
    (
        mobile_path,
        mobile_checkpoint,
        mobile_manifest_path,
        _mobile_manifest,
    ) = M.find_final_checkpoint_and_manifest()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    mobile_model = M.load_model(
        mobile_path,
        mobile_checkpoint,
        device,
    )
    mobile_model.eval()

    layer_names = [
        M.fixed_backbone_target_layer(
            mobile_model,
            mobile_model.image_size,
            resolution,
        )
        for resolution
        in LAYERCAM_RESOLUTIONS
    ]

    target_layers = [
        M.module_by_name(
            mobile_model,
            layer_name,
        )
        for layer_name
        in layer_names
    ]

    mobile_wrapper = (
        M.ClassificationOnlyWrapper(
            mobile_model
        )
        .to(device)
    )
    mobile_wrapper.eval()

    print("MobileNet checkpoint:", mobile_path)
    print("MobileNet training manifest:", mobile_manifest_path)
    print(
        "Layer-CAM target layers:",
        layer_names,
    )

    # ------------------------------------------------------------------
    # YOLO
    # ------------------------------------------------------------------
    yolo_path = (
        find_final_yolo_checkpoint()
    )

    yolo_sha = sha256_file(
        yolo_path
    )

    yolo_model = YOLO(
        str(yolo_path)
    )

    yolo_device = (
        0
        if torch.cuda.is_available()
        else "cpu"
    )

    print("YOLO checkpoint:", yolo_path)
    print("YOLO SHA256:", yolo_sha)
    print(
        "YOLO inference:",
        {
            "imgsz": YOLO_IMAGE_SIZE,
            "conf": YOLO_RETRIEVAL_CONFIDENCE,
            "iou": YOLO_NMS_IOU,
            "max_det": YOLO_MAX_DETECTIONS,
            "augment": YOLO_AUGMENT,
            "retina_masks": True,
        },
    )
    print()

    # ------------------------------------------------------------------
    # Test cohort
    # ------------------------------------------------------------------
    official_train = (
        M.load_official_train_metadata()
    )

    test_frame = (
        M.load_official_test_metadata_after_freeze(
            official_train
        )
        .reset_index(drop=True)
    )

    if len(test_frame) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            "Unexpected test image count."
        )

    if (
        test_frame["patient_id"].nunique()
        != EXPECTED_TEST_PATIENTS
    ):
        raise RuntimeError(
            "Unexpected test patient count."
        )

    dataset = M.AnatomicalXAIDataset(
        test_frame,
        split="test",
        image_size=mobile_model.image_size,
        max_cases=0,
    )

    qualitative_patients = (
        choose_qualitative_patients(
            test_frame
        )
    )

    saved_qualitative_patients = set()

    rows = []

    with LayerCAM(
        model=mobile_wrapper,
        target_layers=target_layers,
    ) as cam_engine:

        for item in tqdm(
            dataset,
            total=len(dataset),
            desc="Layer-CAM / YOLO agreement",
            unit="img",
            dynamic_ncols=True,
        ):
            filename = str(
                item["filename"]
            )
            patient_id = str(
                item["patient_id"]
            )

            image_tensor = (
                item["image"]
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                logit = (
                    mobile_model.classify(
                        image_tensor
                    )
                    .reshape(-1)[0]
                )
                mobile_probability = float(
                    torch.sigmoid(
                        logit
                    ).item()
                )

            mobile_prediction = int(
                mobile_probability
                >= MOBILENET_FRAME_THRESHOLD
            )

            padded_cam = cam_engine(
                input_tensor=image_tensor,
                targets=[
                    M.PredictedBinaryClassTarget(
                        mobile_prediction
                    )
                ],
                aug_smooth=False,
                eigen_smooth=False,
            )[0]

            anatomical_cam, _ = (
                M.inverse_map_cam_to_original(
                    padded_cam,
                    item["valid_region"],
                    item["original_height"],
                    item["original_width"],
                )
            )

            anatomical_cam = normalise_cam(
                anatomical_cam
            )

            # Some evaluator versions expose an original mask as HxW while
            # others can retain a singleton channel axis (HxWx1).  Collapse
            # singleton axes here so every spatial metric receives a strict 2D
            # original-ultrasound mask.
            expert_mask = as_2d_binary_mask(
                item["original_mask"],
                f"expert mask for {filename}",
            )

            image_path = M.fetch(
                f"test/images/{filename}"
            )

            yolo = yolo_top_mask(
                yolo_model,
                image_path,
                (
                    item["original_height"],
                    item["original_width"],
                ),
                yolo_device,
            )

            yolo_mask = as_2d_binary_mask(
                yolo["mask"],
                f"YOLO mask for {filename}",
            )
            if yolo_mask.shape != expert_mask.shape:
                raise RuntimeError(
                    f"Original-geometry mask mismatch for {filename}: "
                    f"YOLO={yolo_mask.shape}, expert={expert_mask.shape}."
                )

            agreement = (
                layercam_yolo_metrics(
                    anatomical_cam,
                    yolo_mask,
                )
            )

            # Layer-CAM vs expert context.
            layer_expert = (
                M.localisation_metrics(
                    anatomical_cam,
                    expert_mask,
                )
            )

            # YOLO vs expert context.
            yolo_expert = binary_mask_metrics(
                yolo_mask,
                expert_mask,
            )

            row = {
                "filename": filename,
                "patient_id": patient_id,
                "label": int(
                    item["label"]
                ),
                "mobile_probability_malignant": (
                    mobile_probability
                ),
                "mobile_prediction": (
                    mobile_prediction
                ),
                "layercam_target_resolutions": (
                    "32|16"
                ),
                "layercam_target_layers": (
                    "|".join(
                        layer_names
                    )
                ),
                "layercam_top_fraction": float(
                    LAYERCAM_TOP_FRACTION
                ),
                "yolo_has_detection": int(
                    yolo["has_detection"]
                ),
                "yolo_n_detections": int(
                    yolo["n_detections"]
                ),
                "yolo_top_confidence": float(
                    yolo["top_confidence"]
                ),
                "yolo_top_class_id": (
                    yolo["top_class_id"]
                ),
                "yolo_top_class_name": (
                    yolo["top_class_name"]
                ),
                "yolo_mask_fraction": float(
                    yolo_mask.mean()
                ),
                "expert_mask_fraction": float(
                    expert_mask.mean()
                ),
                **agreement,
                "layercam_expert_top15_iou": float(
                    layer_expert[
                        "top15_iou"
                    ]
                ),
                "layercam_expert_top15_dice": float(
                    layer_expert[
                        "top15_dice"
                    ]
                ),
                "layercam_expert_overlap_hit_0_5": float(
                    layer_expert[
                        "overlap_hit_0_5"
                    ]
                ),
                "layercam_expert_strict_pointing_hit": float(
                    layer_expert[
                        "strict_pointing_hit"
                    ]
                ),
                "layercam_expert_nodule_energy_fraction": float(
                    layer_expert[
                        "nodule_energy_fraction"
                    ]
                ),
                "yolo_expert_mask_iou": float(
                    yolo_expert[
                        "iou"
                    ]
                ),
                "yolo_expert_mask_dice": float(
                    yolo_expert[
                        "dice"
                    ]
                ),
                "yolo_expert_any_overlap": float(
                    yolo_expert[
                        "any_overlap"
                    ]
                ),
            }

            rows.append(row)

            if (
                patient_id
                in qualitative_patients
                and patient_id
                not in saved_qualitative_patients
            ):
                save_qualitative_panel(
                    filename=filename,
                    patient_id=patient_id,
                    label=int(
                        item["label"]
                    ),
                    image_path=image_path,
                    expert_mask=expert_mask,
                    cam=anatomical_cam,
                    yolo_mask=yolo_mask,
                    row_metrics=row,
                )

                saved_qualitative_patients.add(
                    patient_id
                )

    cases = pd.DataFrame(
        rows
    )

    if len(cases) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            "Agreement case row count mismatch."
        )

    cases_path = (
        OUT
        / "layercam_yolo_agreement_cases.csv"
    )

    cases.to_csv(
        cases_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {
        "status": (
            "THYROIDXL_MOBILENET_LAYERCAM_YOLO_SPATIAL_AGREEMENT"
        ),
        "protocol_version": PROTOCOL_VERSION,
        "official_test_used_for_tuning": False,
        "mobile_checkpoint": str(
            mobile_path
        ),
        "mobile_training_manifest": str(
            mobile_manifest_path
        ),
        "mobile_layercam": {
            "target_resolutions": list(
                LAYERCAM_RESOLUTIONS
            ),
            "target_layers": layer_names,
            "frame_threshold": (
                MOBILENET_FRAME_THRESHOLD
            ),
            "top_fraction": (
                LAYERCAM_TOP_FRACTION
            ),
            "target_policy": (
                "predicted binary class: malignant -> +logit; benign -> -logit"
            ),
        },
        "yolo_checkpoint": str(
            yolo_path
        ),
        "yolo_checkpoint_sha256": (
            yolo_sha
        ),
        "yolo_inference": {
            "image_size": (
                YOLO_IMAGE_SIZE
            ),
            "retrieval_confidence": (
                YOLO_RETRIEVAL_CONFIDENCE
            ),
            "nms_iou": (
                YOLO_NMS_IOU
            ),
            "max_detections": (
                YOLO_MAX_DETECTIONS
            ),
            "augment": (
                YOLO_AUGMENT
            ),
            "retina_masks": True,
            "mask_rule": (
                "highest-confidence retained detection"
            ),
        },
        "comparison_coordinate_space": (
            "original_ultrasound"
        ),
        "overall": (
            patient_balanced_summary(
                cases
            )
        ),
        "patient_bootstrap_ci95": (
            patient_bootstrap_ci(
                cases
            )
        ),
        "by_diagnosis": {},
        "detected_only": None,
    }

    for label, group in cases.groupby(
        "label"
    ):
        summary[
            "by_diagnosis"
        ][
            "malignant"
            if int(label) == 1
            else "benign"
        ] = patient_balanced_summary(
            group
        )

    detected = cases[
        cases[
            "yolo_has_detection"
        ]
        == 1
    ].copy()

    if len(detected):
        summary[
            "detected_only"
        ] = {
            "summary": (
                patient_balanced_summary(
                    detected
                )
            ),
            "patient_bootstrap_ci95": (
                patient_bootstrap_ci(
                    detected,
                    seed=SEED + 100,
                )
            ),
        }

    # Per-patient compact table.
    patient_metrics = (
        cases.groupby(
            [
                "patient_id",
                "label",
            ],
            as_index=False,
        )[
            list(
                SUMMARY_METRICS
            )
            + [
                "yolo_has_detection",
            ]
        ]
        .mean()
    )

    patient_path = (
        OUT
        / "layercam_yolo_agreement_patients.csv"
    )

    patient_metrics.to_csv(
        patient_path,
        index=False,
    )

    summary_path = (
        OUT
        / "layercam_yolo_agreement_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Compact publication-ready summary CSV.
    compact_rows = []

    groups = [
        ("Overall", cases),
        (
            "Benign",
            cases[cases["label"] == 0],
        ),
        (
            "Malignant",
            cases[cases["label"] == 1],
        ),
        (
            "YOLO detected only",
            detected,
        ),
    ]

    for group_name, group in groups:
        if len(group) == 0:
            continue

        record = patient_balanced_summary(
            group
        )

        compact_rows.append(
            {
                "group": group_name,
                **record,
            }
        )

    compact = pd.DataFrame(
        compact_rows
    )

    compact_path = (
        OUT
        / "layercam_yolo_agreement_summary.csv"
    )

    compact.to_csv(
        compact_path,
        index=False,
    )

    overall = summary["overall"]

    print()
    print("=" * 80)
    print("FINAL SPATIAL AGREEMENT SUMMARY")
    print("=" * 80)
    print(
        "YOLO detection coverage: "
        f"{overall['yolo_detection_coverage'] * 100.0:.1f}%"
    )
    print(
        "Layer-CAM top15 vs YOLO IoU: "
        f"{overall['patient_balanced_mean_layercam_yolo_top15_iou']:.4f}"
    )
    print(
        "Layer-CAM top15 vs YOLO Dice: "
        f"{overall['patient_balanced_mean_layercam_yolo_top15_dice']:.4f}"
    )
    print(
        "Any spatial overlap: "
        f"{overall['patient_balanced_mean_layercam_yolo_any_overlap'] * 100.0:.1f}%"
    )
    print(
        "Layer-CAM peak inside YOLO mask: "
        f"{overall['patient_balanced_mean_layercam_peak_inside_yolo'] * 100.0:.1f}%"
    )
    print(
        "Layer-CAM energy inside YOLO mask: "
        f"{overall['patient_balanced_mean_layercam_energy_inside_yolo']:.4f}"
    )
    print()
    print(
        "Layer-CAM vs expert IoU: "
        f"{overall['patient_balanced_mean_layercam_expert_top15_iou']:.4f}"
    )
    print(
        "Layer-CAM peak inside expert mask: "
        f"{overall['patient_balanced_mean_layercam_expert_strict_pointing_hit'] * 100.0:.1f}%"
    )
    print(
        "YOLO vs expert mask IoU: "
        f"{overall['patient_balanced_mean_yolo_expert_mask_iou']:.4f}"
    )
    print()
    print("Saved:")
    print("  Cases:", cases_path)
    print("  Patients:", patient_path)
    print("  Summary CSV:", compact_path)
    print("  Summary JSON:", summary_path)
    print("  Qualitative panels:", QUAL_OUT)


if __name__ == "__main__":
    main()
