from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from huggingface_hub import get_token, hf_hub_download
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Frozen study constants — matched to the training notebook
# ---------------------------------------------------------------------

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO_ID = "hunglc007/ThyroidXL"
REPO_REVISION = "b15fe293bd74f1a8a4f05bf88bcdf06a1934125f"

EXPECTED_TRAIN_IMAGES = 9541
EXPECTED_TRAIN_PATIENTS = 3354
EXPECTED_TRAIN_BENIGN_PATIENTS = 2477
EXPECTED_TRAIN_MALIGNANT_PATIENTS = 877

EXPECTED_TEST_IMAGES = 2094
EXPECTED_TEST_PATIENTS = 739
EXPECTED_TEST_BENIGN_PATIENTS = 386
EXPECTED_TEST_MALIGNANT_PATIENTS = 353

EXPECTED_FOLD1_TRAIN_IMAGES = 7684
EXPECTED_FOLD1_TRAIN_PATIENTS = 2683
EXPECTED_FOLD1_VAL_IMAGES = 1857
EXPECTED_FOLD1_VAL_PATIENTS = 671

FINAL_STATUS = "THYROIDXL_CONVNEXTTINY_FINAL_OFFICIAL_TRAIN_REFIT_ONEFOLD_SELECTED"
DEVELOPMENT_STATUS = "THYROIDXL_CONVNEXTTINY_ONEFOLD_DEVELOPMENT_SELECTED"

MODEL_VARIANT = "ConvNeXtTiny_Multiscale_DiceBCE"
DEFAULT_MODEL_NAME = "convnext_tiny"
DEFAULT_IMAGE_SIZE = 512
DEFAULT_DROP_RATE = 0.20

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

PRIMARY_IMAGE_THRESHOLD = 0.50
PRIMARY_PATIENT_THRESHOLD = 0.50

SEED = 42
N_FOLDS = 5
FOLD_INDEX = 1

SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------
# Project / model discovery
# ---------------------------------------------------------------------

def project_root() -> Path:
    env = os.environ.get("THYROIDXL_PROJECT_ROOT")
    if env:
        candidate = Path(env).expanduser().resolve()
        if candidate.is_dir():
            return candidate

    for candidate in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (candidate / "Models").is_dir():
            return candidate

    raise RuntimeError(
        "Could not locate the ThyroidXL project root. Expected a Models/ "
        "directory above this script, or set THYROIDXL_PROJECT_ROOT."
    )


ROOT = project_root()
CONVNEXT_MODELS_ROOT = ROOT / "Models" / "ConvNeXtTiny"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_load_compatible(path: Path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def find_final_checkpoint_and_manifest():
    if not CONVNEXT_MODELS_ROOT.is_dir():
        raise FileNotFoundError(
            f"Missing ConvNeXt-Tiny model directory: {CONVNEXT_MODELS_ROOT}"
        )

    checkpoint_matches = []

    for path in CONVNEXT_MODELS_ROOT.rglob("*.pt"):
        try:
            payload = torch_load_compatible(path, "cpu")
        except Exception:
            continue

        if (
            isinstance(payload, dict)
            and payload.get("status") == FINAL_STATUS
        ):
            checkpoint_matches.append((path.resolve(), payload))

    if len(checkpoint_matches) != 1:
        raise RuntimeError(
            "Expected exactly one FINAL checkpoint produced by the reviewed "
            f"training notebook (status={FINAL_STATUS!r}). Found "
            f"{len(checkpoint_matches)}:\n"
            + "\n".join(str(path) for path, _ in checkpoint_matches)
        )

    checkpoint_path, checkpoint = checkpoint_matches[0]

    required = {
        "state_dict",
        "status",
        "publication_role",
        "dataset",
        "dataset_repo",
        "dataset_revision",
        "official_test_accessed",
        "internal_validation_used_in_final_refit",
        "model_variant",
        "model_name",
        "image_size",
        "drop_rate",
        "training_images",
        "training_patients",
        "selection_source",
        "primary_patient_threshold",
        "development_selected_patient_threshold",
        "validation_patient_threshold",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(
            f"Final checkpoint is missing required metadata: {missing}"
        )

    if checkpoint["dataset"] != "ThyroidXL":
        raise RuntimeError("Final checkpoint dataset is not ThyroidXL.")
    if checkpoint["dataset_repo"] != REPO_ID:
        raise RuntimeError("Final checkpoint repository mismatch.")
    if checkpoint["dataset_revision"] != REPO_REVISION:
        raise RuntimeError("Final checkpoint dataset revision mismatch.")
    if checkpoint["model_variant"] != MODEL_VARIANT:
        raise RuntimeError(
            f"Expected model_variant={MODEL_VARIANT!r}, got "
            f"{checkpoint['model_variant']!r}."
        )
    if int(checkpoint["training_images"]) != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError(
            "Final checkpoint was not trained on all 9,541 official-training images."
        )
    if int(checkpoint["training_patients"]) != EXPECTED_TRAIN_PATIENTS:
        raise RuntimeError(
            "Final checkpoint was not trained on all 3,354 official-training patients."
        )
    if checkpoint["official_test_accessed"] is not False:
        raise RuntimeError(
            "Final training checkpoint does not certify untouched official test."
        )
    if checkpoint["internal_validation_used_in_final_refit"] is not False:
        raise RuntimeError(
            "Final checkpoint says internal validation was used during final refit."
        )

    selection = checkpoint["selection_source"]
    if int(selection.get("validation_images", -1)) != EXPECTED_FOLD1_VAL_IMAGES:
        raise RuntimeError("Final checkpoint selection-source validation image count mismatch.")
    if int(selection.get("validation_patients", -1)) != EXPECTED_FOLD1_VAL_PATIENTS:
        raise RuntimeError("Final checkpoint selection-source validation patient count mismatch.")
    if int(selection.get("patient_overlap", -1)) != 0:
        raise RuntimeError("Final checkpoint selection source reports patient overlap.")

    primary_threshold = float(checkpoint["primary_patient_threshold"])
    if not np.isclose(primary_threshold, PRIMARY_PATIENT_THRESHOLD, atol=1e-12, rtol=0):
        raise RuntimeError(
            "Final checkpoint primary patient threshold is not the reviewed 0.5 rule."
        )

    development_threshold = float(
        checkpoint["development_selected_patient_threshold"]
    )
    validation_alias = float(checkpoint["validation_patient_threshold"])
    if not np.isclose(
        development_threshold,
        validation_alias,
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError(
            "Final checkpoint development threshold and compatibility alias disagree."
        )

    checkpoint_sha = sha256_file(checkpoint_path)

    manifest_matches = []
    for path in ROOT.rglob("*_training_manifest.json"):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if (
            payload.get("status") == FINAL_STATUS
            and payload.get("checkpoint_sha256") == checkpoint_sha
        ):
            manifest_matches.append((path.resolve(), payload))

    if len(manifest_matches) != 1:
        raise RuntimeError(
            "Expected exactly one final *_training_manifest.json whose "
            f"checkpoint_sha256 matches {checkpoint_sha}. Found "
            f"{len(manifest_matches)}."
        )

    manifest_path, manifest = manifest_matches[0]

    if manifest.get("official_test_accessed") is not False:
        raise RuntimeError(
            "Final training manifest does not certify untouched official test."
        )
    if int(manifest.get("training_images", EXPECTED_TRAIN_IMAGES)) != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError("Final ConvNeXt-Tiny manifest training-image count mismatch.")
    if int(manifest.get("training_patients", EXPECTED_TRAIN_PATIENTS)) != EXPECTED_TRAIN_PATIENTS:
        raise RuntimeError("Final ConvNeXt-Tiny manifest training-patient count mismatch.")

    return checkpoint_path, checkpoint, manifest_path, manifest


def find_development_checkpoint(
    expected_sha256: str | None = None,
):
    if not CONVNEXT_MODELS_ROOT.is_dir():
        raise FileNotFoundError(
            f"Missing ConvNeXt-Tiny model directory: {CONVNEXT_MODELS_ROOT}"
        )

    matches = []

    for path in CONVNEXT_MODELS_ROOT.rglob("*.pt"):
        try:
            payload = torch_load_compatible(path, "cpu")
        except Exception:
            continue

        if (
            isinstance(payload, dict)
            and payload.get("status") == DEVELOPMENT_STATUS
        ):
            current_sha = sha256_file(path)
            if (
                expected_sha256 is not None
                and current_sha != expected_sha256
            ):
                continue
            matches.append((path.resolve(), payload))

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one reviewed DEVELOPMENT checkpoint. "
            f"Found {len(matches)}:\n"
            + "\n".join(str(path) for path, _ in matches)
        )

    path, checkpoint = matches[0]

    if int(checkpoint.get("training_images", -1)) != EXPECTED_FOLD1_TRAIN_IMAGES:
        raise RuntimeError("Development checkpoint training image count mismatch.")
    if int(checkpoint.get("training_patients", -1)) != EXPECTED_FOLD1_TRAIN_PATIENTS:
        raise RuntimeError("Development checkpoint training patient count mismatch.")
    if int(checkpoint.get("validation_images", -1)) != EXPECTED_FOLD1_VAL_IMAGES:
        raise RuntimeError("Development checkpoint validation image count mismatch.")
    if int(checkpoint.get("validation_patients", -1)) != EXPECTED_FOLD1_VAL_PATIENTS:
        raise RuntimeError("Development checkpoint validation patient count mismatch.")
    if checkpoint.get("official_test_accessed") is not False:
        raise RuntimeError("Development checkpoint does not certify untouched official test.")

    return path, checkpoint


# ---------------------------------------------------------------------
# Hugging Face access
# ---------------------------------------------------------------------

def resolve_hf_token() -> str:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or get_token()
    )
    if not token:
        raise RuntimeError(
            "ThyroidXL is gated. Run `hf auth login` once, or set HF_TOKEN."
        )
    return str(token).strip()


HF_TOKEN = resolve_hf_token()


def fetch(repo_path: str, max_attempts: int = 20) -> Path:
    repo_path = str(repo_path).replace("\\", "/").lstrip("/")

    try:
        return Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_path,
                repo_type="dataset",
                revision=REPO_REVISION,
                token=HF_TOKEN,
                local_files_only=True,
            )
        ).resolve()
    except Exception:
        pass

    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            return Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=repo_path,
                    repo_type="dataset",
                    revision=REPO_REVISION,
                    token=HF_TOKEN,
                    local_files_only=False,
                )
            ).resolve()
        except Exception as exc:
            last_error = exc
            text = repr(exc).lower()

            if "404" in text:
                raise FileNotFoundError(repo_path) from exc
            if "401" in text or "403" in text:
                raise PermissionError(
                    f"Hugging Face denied access to {repo_path}."
                ) from exc

            if attempt == max_attempts:
                break

            wait = min(180.0, 5.0 * attempt)
            print(
                f"Hugging Face access error for {repo_path}: {exc}\n"
                f"Retrying in {wait:.0f}s ({attempt + 1}/{max_attempts})..."
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Could not fetch {repo_path} after {max_attempts} attempts."
    ) from last_error


# ---------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------

def patient_id_from_filename(filename: str) -> str:
    stem = Path(str(filename)).stem
    match = re.match(r"^(\d+)(?:_|$)", stem)
    if match is None:
        raise ValueError(
            f"Cannot derive patient ID from filename: {filename}"
        )
    return str(int(match.group(1)))


def category_to_binary(category_id, mapping: dict):
    name = str(mapping.get(category_id, "")).strip().lower()

    if "benign" in name:
        return 0
    if "malignant" in name:
        return 1

    if category_id in (0, 1):
        return int(category_id)
    if str(category_id).strip() in {"0", "1"}:
        return int(category_id)

    return None


def build_coco_frame(path: Path) -> pd.DataFrame:
    with Path(path).open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    for key in ("images", "annotations"):
        if key not in coco:
            raise RuntimeError(
                f"{path.name} has no {key!r} key."
            )

    mapping = {
        item["id"]: str(item.get("name", "")).strip()
        for item in coco.get("categories", [])
        if isinstance(item, dict) and "id" in item
    }

    rows = []
    image_id_to_filename = {}

    for item in coco["images"]:
        image_id = item["id"]
        filename = Path(str(item["file_name"])).name

        if image_id in image_id_to_filename:
            raise RuntimeError(
                f"Duplicate COCO image ID: {image_id}"
            )

        image_id_to_filename[image_id] = filename
        rows.append(
            {
                "image_id": image_id,
                "filename": filename,
                "patient_id": patient_id_from_filename(filename),
            }
        )

    categories_by_image = {}

    for annotation in coco["annotations"]:
        if not isinstance(annotation, dict):
            continue

        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")

        if (
            image_id in image_id_to_filename
            and category_id is not None
        ):
            categories_by_image.setdefault(
                image_id,
                set(),
            ).add(category_id)

    labels = {}
    failures = []

    for image_id, filename in image_id_to_filename.items():
        values = {
            category_to_binary(category_id, mapping)
            for category_id in categories_by_image.get(
                image_id,
                set(),
            )
        }
        values.discard(None)

        if len(values) != 1:
            failures.append(
                {
                    "filename": filename,
                    "category_ids": sorted(
                        map(
                            str,
                            categories_by_image.get(
                                image_id,
                                set(),
                            ),
                        )
                    ),
                    "binary_values": sorted(values),
                }
            )
        else:
            labels[image_id] = next(iter(values))

    if len(labels) != len(rows):
        raise RuntimeError(
            "Could not derive exactly one benign/malignant label for every "
            f"image. Resolved={len(labels)}/{len(rows)}; "
            f"examples={failures[:5]}"
        )

    frame = pd.DataFrame(rows)
    frame["label"] = (
        frame["image_id"].map(labels).astype(int)
    )

    if frame["filename"].duplicated().any():
        raise RuntimeError(
            f"Duplicate filenames in {path.name}."
        )

    consistency = (
        frame.groupby("patient_id")["label"].nunique()
    )
    if int(consistency.max()) != 1:
        raise RuntimeError(
            f"A patient in {path.name} has inconsistent diagnosis labels."
        )

    return (
        frame.sort_values(["patient_id", "filename"])
        .reset_index(drop=True)
    )


def load_official_train_metadata() -> pd.DataFrame:
    train = build_coco_frame(
        fetch("train/train_annotations.json")
    )

    if len(train) != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_IMAGES} training images, "
            f"got {len(train)}."
        )

    if (
        train["patient_id"].nunique()
        != EXPECTED_TRAIN_PATIENTS
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_PATIENTS} training patients, "
            f"got {train['patient_id'].nunique()}."
        )

    patients = (
        train[["patient_id", "label"]]
        .drop_duplicates()
    )

    counts = (
        patients["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    expected = {
        0: EXPECTED_TRAIN_BENIGN_PATIENTS,
        1: EXPECTED_TRAIN_MALIGNANT_PATIENTS,
    }

    if counts != expected:
        raise RuntimeError(
            "Official training patient class counts mismatch. "
            f"Expected={expected}, observed={counts}"
        )

    return train


def load_official_test_metadata_after_freeze(
    train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Open the official held-out split only after all development/freeze steps
    required by the calling script have completed.
    """
    test = build_coco_frame(
        fetch("test/test_annotations.json")
    )

    if len(test) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_IMAGES} held-out images, "
            f"got {len(test)}."
        )

    if (
        test["patient_id"].nunique()
        != EXPECTED_TEST_PATIENTS
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_PATIENTS} held-out patients, "
            f"got {test['patient_id'].nunique()}."
        )

    patients = (
        test[["patient_id", "label"]]
        .drop_duplicates()
    )

    counts = (
        patients["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    expected = {
        0: EXPECTED_TEST_BENIGN_PATIENTS,
        1: EXPECTED_TEST_MALIGNANT_PATIENTS,
    }

    if counts != expected:
        raise RuntimeError(
            "Official held-out patient class counts mismatch. "
            f"Expected={expected}, observed={counts}"
        )

    patient_overlap = (
        set(train["patient_id"])
        & set(test["patient_id"])
    )
    filename_overlap = (
        set(train["filename"])
        & set(test["filename"])
    )

    if patient_overlap:
        raise RuntimeError(
            "Official train/test PATIENT overlap detected: "
            f"{sorted(patient_overlap)[:10]}"
        )

    if filename_overlap:
        raise RuntimeError(
            "Official train/test FILENAME overlap detected: "
            f"{sorted(filename_overlap)[:10]}"
        )

    return test


def reconstruct_fold1_validation(
    train: pd.DataFrame,
) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedKFold

    patients = (
        train[["patient_id", "label"]]
        .drop_duplicates()
        .sort_values(
            "patient_id",
            key=lambda series: series.astype(int),
        )
        .reset_index(drop=True)
    )
    patients["fold"] = -1

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    for zero_fold, (_, val_indices) in enumerate(
        splitter.split(
            patients["patient_id"],
            patients["label"],
        )
    ):
        patients.loc[
            val_indices,
            "fold",
        ] = zero_fold + 1

    patient_to_fold = (
        patients.set_index("patient_id")["fold"]
        .to_dict()
    )

    with_fold = train.copy()
    with_fold["fold"] = (
        with_fold["patient_id"]
        .map(patient_to_fold)
        .astype(int)
    )

    fold_train = (
        with_fold[with_fold["fold"] != FOLD_INDEX]
        .sort_values(["patient_id", "filename"])
        .reset_index(drop=True)
    )
    fold_val = (
        with_fold[with_fold["fold"] == FOLD_INDEX]
        .sort_values(["patient_id", "filename"])
        .reset_index(drop=True)
    )

    if len(fold_train) != EXPECTED_FOLD1_TRAIN_IMAGES:
        raise RuntimeError(
            "Reconstructed Fold-1 training image count mismatch."
        )
    if (
        fold_train["patient_id"].nunique()
        != EXPECTED_FOLD1_TRAIN_PATIENTS
    ):
        raise RuntimeError(
            "Reconstructed Fold-1 training patient count mismatch."
        )
    if len(fold_val) != EXPECTED_FOLD1_VAL_IMAGES:
        raise RuntimeError(
            "Reconstructed Fold-1 validation image count mismatch."
        )
    if (
        fold_val["patient_id"].nunique()
        != EXPECTED_FOLD1_VAL_PATIENTS
    ):
        raise RuntimeError(
            "Reconstructed Fold-1 validation patient count mismatch."
        )

    overlap = (
        set(fold_train["patient_id"])
        & set(fold_val["patient_id"])
    )
    if overlap:
        raise RuntimeError(
            "Patient leakage detected in reconstructed Fold 1."
        )

    return fold_val


# ---------------------------------------------------------------------
# Exact ConvNeXt-Tiny + segmentation decoder reconstruction
# ---------------------------------------------------------------------

class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
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
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.refine = ConvBNAct(
            in_channels,
            out_channels,
        )

    def forward(self, x, target_size):
        x = F.interpolate(
            x,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.refine(x)


class ConvNeXtTinyMultiscaleDiceBCE(nn.Module):
    """Exact reconstruction of the reviewed ConvNeXt-Tiny multitask network."""
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        image_size: int = DEFAULT_IMAGE_SIZE,
        drop_rate: float = DEFAULT_DROP_RATE,
        expected_feature_channels: dict | None = None,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=1,
            drop_rate=float(drop_rate),
        )

        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            final_feature, skips = self._encode(dummy)
        if was_training:
            self.backbone.train()

        inferred = {
            "final": int(final_feature.shape[1]),
            "skip32": int(skips[32].shape[1]),
            "skip64": int(skips[64].shape[1]),
            "skip128": int(skips[128].shape[1]),
        }
        if expected_feature_channels is not None:
            expected = {str(k): int(v) for k, v in expected_feature_channels.items()}
            if inferred != expected:
                raise RuntimeError(
                    "Installed timm ConvNeXt-Tiny architecture does not match "
                    f"checkpoint metadata. Expected={expected}, inferred={inferred}"
                )

        expected_final = (self.image_size // 32, self.image_size // 32)
        if tuple(final_feature.shape[-2:]) != expected_final:
            raise RuntimeError(
                "Unexpected final ConvNeXt-Tiny feature resolution: "
                f"{tuple(final_feature.shape)}"
            )

        self.decoder_bottleneck = ConvBNAct(inferred["final"], 192)
        self.decoder_skip32 = SkipFusionBlock(192, inferred["skip32"], 128)
        self.decoder_skip64 = SkipFusionBlock(128, inferred["skip64"], 96)
        self.decoder_skip128 = SkipFusionBlock(96, inferred["skip128"], 64)
        self.decoder_up256 = UpsampleRefineBlock(64, 32)
        self.decoder_up512 = UpsampleRefineBlock(32, 16)
        self.segmentation_output = nn.Conv2d(16, 1, kernel_size=1)

    def _encode(self, image):
        x = self.backbone.stem(image)
        stage_outputs = []
        for stage in self.backbone.stages:
            x = stage(x)
            stage_outputs.append(x)

        skips = {}
        for target in (32, 64, 128):
            candidates = [
                feature for feature in stage_outputs
                if tuple(feature.shape[-2:]) == (target, target)
            ]
            if not candidates:
                available = sorted({tuple(feature.shape[-2:]) for feature in stage_outputs})
                raise RuntimeError(
                    f"No ConvNeXt-Tiny skip feature at {target}x{target}. "
                    f"Available={available}"
                )
            skips[target] = candidates[-1]

        x = self.backbone.norm_pre(x)
        return x, skips

    def classify(self, image):
        final_feature, _ = self._encode(image)
        return self.backbone.forward_head(final_feature).flatten()

    def forward(self, image):
        final_feature, skips = self._encode(image)
        classification_logits = self.backbone.forward_head(final_feature).flatten()

        x = self.decoder_bottleneck(final_feature)
        x = self.decoder_skip32(x, skips[32])
        x = self.decoder_skip64(x, skips[64])
        x = self.decoder_skip128(x, skips[128])
        x = self.decoder_up256(x, (self.image_size // 2, self.image_size // 2))
        x = self.decoder_up512(x, (self.image_size, self.image_size))
        segmentation_logits = self.segmentation_output(x)

        if segmentation_logits.shape[-2:] != image.shape[-2:]:
            segmentation_logits = F.interpolate(
                segmentation_logits,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return classification_logits, segmentation_logits


def load_model(
    checkpoint_path: Path,
    checkpoint: dict,
    device: torch.device,
):
    model = ConvNeXtTinyMultiscaleDiceBCE(
        model_name=str(checkpoint.get("model_name", DEFAULT_MODEL_NAME)),
        image_size=int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE)),
        drop_rate=float(checkpoint.get("drop_rate", DEFAULT_DROP_RATE)),
        expected_feature_channels=checkpoint.get("feature_channels"),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------
# Evaluation transform / dataset
# ---------------------------------------------------------------------

def make_eval_transform(image_size: int):
    """
    Exact final-model preprocessing plus a valid-region mask.

    ``valid_region`` is 1 on pixels originating from the real ultrasound and
    0 on synthetic square padding.  It is NEVER provided to the model; it is
    retained only so the auxiliary segmentation probability map can be mapped
    back to the original ultrasound coordinate system after inference.
    """
    return A.Compose(
        [
            A.LongestMaxSize(
                max_size=int(image_size),
                area_for_downscale="image",
            ),
            A.PadIfNeeded(
                min_height=int(image_size),
                min_width=int(image_size),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
            ),
            A.Normalize(
                mean=IMAGE_MEAN,
                std=IMAGE_STD,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        additional_targets={
            "valid_region": "mask",
        },
        seed=SEED,
        strict=True,
    )


class ThyroidXLImageMaskDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        split: str,
        image_size: int,
    ):
        self.frame = frame.reset_index(drop=True)
        self.split = str(split)
        self.transform = make_eval_transform(
            int(image_size)
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        filename = str(row["filename"])

        image_path = fetch(
            f"{self.split}/images/{filename}"
        )
        mask_path = fetch(
            f"{self.split}/masks/{filename}"
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )
        if mask is None:
            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        if image.shape[:2] != mask.shape[:2]:
            raise RuntimeError(
                f"Image/mask shape mismatch for {filename}: "
                f"{image.shape[:2]} vs {mask.shape[:2]}"
            )

        mask = (mask > 0).astype(np.uint8)

        if not mask.any():
            raise RuntimeError(
                f"Empty expert nodule mask: {filename}"
            )

        original_height = int(mask.shape[0])
        original_width = int(mask.shape[1])
        valid_region = np.ones_like(
            mask,
            dtype=np.uint8,
        )

        transformed = self.transform(
            image=image,
            mask=mask,
            valid_region=valid_region,
        )

        image_tensor = transformed["image"].float()
        mask_tensor = transformed["mask"]
        valid_region_tensor = transformed["valid_region"]

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        if valid_region_tensor.ndim == 2:
            valid_region_tensor = valid_region_tensor.unsqueeze(0)

        mask_tensor = (
            mask_tensor > 0
        ).float()
        valid_region_tensor = (
            valid_region_tensor > 0
        ).to(torch.uint8)

        if torch.any(
            (mask_tensor > 0)
            & (valid_region_tensor == 0)
        ):
            raise RuntimeError(
                "Transformed expert mask overlaps synthetic padding: "
                f"{filename}"
            )

        return {
            "image": image_tensor,
            # Retained for transparent secondary padded-space metrics.
            "mask": mask_tensor,
            # Used only to undo the preprocessing geometry after inference.
            "valid_region": valid_region_tensor,
            "original_height": original_height,
            "original_width": original_width,
            "mask_path": str(mask_path),
            "filename": filename,
            "patient_id": str(row["patient_id"]),
            "label": torch.tensor(
                int(row["label"]),
                dtype=torch.long,
            ),
        }


# ---------------------------------------------------------------------
# Aggregation / metrics
# ---------------------------------------------------------------------

def hard_segmentation_metrics(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
):
    probabilities = torch.sigmoid(logits)
    predictions = (
        probabilities >= threshold
    ).float()

    dims = (1, 2, 3)
    intersection = (
        predictions * masks
    ).sum(dim=dims)

    pred_sum = predictions.sum(dim=dims)
    target_sum = masks.sum(dim=dims)

    dice = (
        2.0 * intersection + 1e-6
    ) / (
        pred_sum + target_sum + 1e-6
    )

    iou = (
        intersection + 1e-6
    ) / (
        pred_sum
        + target_sum
        - intersection
        + 1e-6
    )

    return {
        "dice": dice,
        "iou": iou,
        "predicted_fraction": (
            predictions.mean(dim=dims)
        ),
        "expert_fraction": (
            masks.mean(dim=dims)
        ),
    }


def valid_region_bounds(valid_region):
    """Return the exact non-padding rectangle in model-input coordinates."""
    valid = np.asarray(valid_region).squeeze() > 0

    if valid.ndim != 2 or not valid.any():
        raise ValueError(
            "valid_region must be a non-empty 2-D binary mask."
        )

    rows, cols = np.where(valid)
    y0 = int(rows.min())
    y1 = int(rows.max()) + 1
    x0 = int(cols.min())
    x1 = int(cols.max()) + 1

    rectangle = np.zeros_like(valid, dtype=bool)
    rectangle[y0:y1, x0:x1] = True
    if not np.array_equal(valid, rectangle):
        raise RuntimeError(
            "Expected preprocessing valid region to be one rectangle."
        )

    return y0, y1, x0, x1


def original_geometry_segmentation_metrics(
    padded_probability,
    valid_region,
    mask_path,
    original_height,
    original_width,
    threshold=0.5,
):
    """
    Compute ConvNeXt-Tiny auxiliary-segmentation metrics in ORIGINAL ultrasound
    coordinates.

    The network prediction is produced on the exact padded 512x512 model input.
    After inference we remove ONLY synthetic preprocessing padding, resize the
    probability map back to the original image size, apply the frozen 0.5 mask
    threshold, and compare with the untouched original expert mask.
    """
    probability = np.asarray(
        padded_probability,
        dtype=np.float32,
    ).squeeze()
    valid = np.asarray(valid_region).squeeze() > 0

    if probability.shape != valid.shape:
        raise RuntimeError(
            "Segmentation probability/valid-region shape mismatch: "
            f"{probability.shape} vs {valid.shape}"
        )

    y0, y1, x0, x1 = valid_region_bounds(valid)
    cropped_probability = probability[y0:y1, x0:x1]

    if cropped_probability.size == 0:
        raise RuntimeError(
            "Removing preprocessing padding produced an empty prediction."
        )

    original_height = int(original_height)
    original_width = int(original_width)

    original_probability = cv2.resize(
        cropped_probability,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )

    prediction = (
        original_probability >= float(threshold)
    )

    expert = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )
    if expert is None:
        raise RuntimeError(
            f"Could not reread original expert mask: {mask_path}"
        )
    expert = expert > 0

    if expert.shape != (
        original_height,
        original_width,
    ):
        raise RuntimeError(
            "Original expert-mask geometry mismatch: "
            f"expected={(original_height, original_width)}, "
            f"observed={expert.shape}"
        )

    intersection = float(
        np.logical_and(prediction, expert).sum()
    )
    pred_sum = float(prediction.sum())
    target_sum = float(expert.sum())
    union = pred_sum + target_sum - intersection

    dice = float(
        (2.0 * intersection + 1e-6)
        / (pred_sum + target_sum + 1e-6)
    )
    iou = float(
        (intersection + 1e-6)
        / (union + 1e-6)
    )

    return {
        "dice": dice,
        "iou": iou,
        "predicted_fraction": float(prediction.mean()),
        "expert_fraction": float(expert.mean()),
        "valid_y0": y0,
        "valid_y1": y1,
        "valid_x0": x0,
        "valid_x1": x1,
    }


def aggregate_patient_mean(
    image_df: pd.DataFrame,
):
    consistency = (
        image_df.groupby("patient_id")["label"]
        .nunique()
    )

    if int(consistency.max()) != 1:
        raise RuntimeError(
            "Patient labels are inconsistent."
        )

    return (
        image_df.groupby(
            "patient_id",
            as_index=False,
        )
        .agg(
            label=("label", "first"),
            probability_malignant=(
                "probability_malignant",
                "mean",
            ),
            n_frames=("filename", "size"),
        )
    )


def aggregate_patient_wmv(
    image_df: pd.DataFrame,
    image_threshold: float = 0.5,
):
    """
    Confidence-weighted majority vote:
    - frame classified malignant -> vote weight = p
    - frame classified benign    -> vote weight = 1-p
    - patient class = larger total confidence-weighted vote
    """
    frame = image_df[
        [
            "patient_id",
            "label",
            "probability_malignant",
        ]
    ].copy()

    probability = (
        frame["probability_malignant"]
        .to_numpy(dtype=float)
    )

    malignant_vote = (
        probability >= float(image_threshold)
    )

    frame["benign_vote_weight"] = np.where(
        malignant_vote,
        0.0,
        1.0 - probability,
    )
    frame["malignant_vote_weight"] = np.where(
        malignant_vote,
        probability,
        0.0,
    )

    patient = (
        frame.groupby(
            "patient_id",
            as_index=False,
        )
        .agg(
            label=("label", "first"),
            benign_vote_weight=(
                "benign_vote_weight",
                "sum",
            ),
            malignant_vote_weight=(
                "malignant_vote_weight",
                "sum",
            ),
            n_frames=(
                "probability_malignant",
                "size",
            ),
        )
    )

    patient["prediction_wmv"] = (
        patient["malignant_vote_weight"]
        > patient["benign_vote_weight"]
    ).astype(int)

    ties = (
        patient["malignant_vote_weight"]
        == patient["benign_vote_weight"]
    )

    if ties.any():
        mean_probability = (
            frame.groupby("patient_id")[
                "probability_malignant"
            ].mean()
        )
        patient.loc[
            ties,
            "prediction_wmv",
        ] = (
            patient.loc[ties, "patient_id"]
            .map(mean_probability)
            .ge(0.5)
            .astype(int)
        )

    return patient


def specificity_score(
    labels,
    predictions,
):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(
        predictions,
        dtype=int,
    )

    tn = int(
        (
            (labels == 0)
            & (predictions == 0)
        ).sum()
    )
    fp = int(
        (
            (labels == 0)
            & (predictions == 1)
        ).sum()
    )

    return (
        float(tn / (tn + fp))
        if (tn + fp)
        else float("nan")
    )


def safe_roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if np.unique(labels).size < 2:
        return float("nan")

    return float(
        roc_auc_score(labels, scores)
    )


def safe_auprc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if np.unique(labels).size < 2:
        return float("nan")

    return float(
        average_precision_score(
            labels,
            scores,
        )
    )


def classification_metrics(
    labels,
    probabilities,
    threshold: float,
):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities >= float(threshold)
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "n": int(len(labels)),
        "roc_auc": safe_roc_auc(
            labels,
            probabilities,
        ),
        "auprc": safe_auprc(
            labels,
            probabilities,
        ),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "sensitivity_malignant": float(
            recall_score(
                labels,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "specificity_benign": (
            specificity_score(
                labels,
                predictions,
            )
        ),
        "precision_malignant": float(
            precision_score(
                labels,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1_malignant": float(
            f1_score(
                labels,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                labels,
                predictions,
            )
        ),
        "brier": float(
            brier_score_loss(
                labels,
                probabilities,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def hard_prediction_metrics(
    labels,
    predictions,
):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(
        predictions,
        dtype=int,
    )

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "n": int(len(labels)),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "sensitivity_malignant": float(
            recall_score(
                labels,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "specificity_benign": (
            specificity_score(
                labels,
                predictions,
            )
        ),
        "precision_malignant": float(
            precision_score(
                labels,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1_malignant": float(
            f1_score(
                labels,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                labels,
                predictions,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def expected_calibration_error(
    labels,
    probabilities,
    bins: int = 10,
):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0

    for index in range(bins):
        if index == bins - 1:
            mask = (
                (probabilities >= edges[index])
                & (
                    probabilities
                    <= edges[index + 1]
                )
            )
        else:
            mask = (
                (probabilities >= edges[index])
                & (
                    probabilities
                    < edges[index + 1]
                )
            )

        if not mask.any():
            continue

        confidence = float(
            probabilities[mask].mean()
        )
        observed = float(
            labels[mask].mean()
        )

        ece += (
            float(mask.mean())
            * abs(confidence - observed)
        )

    return float(ece)


def patient_bootstrap_ci(
    patient_df: pd.DataFrame,
    metric: str,
    threshold: float,
    samples: int = 2000,
    seed: int = SEED,
):
    labels_all = (
        patient_df["label"]
        .to_numpy(dtype=int)
    )
    scores_all = (
        patient_df["probability_malignant"]
        .to_numpy(dtype=float)
    )

    rng = np.random.default_rng(seed)
    values = []

    for _ in range(int(samples)):
        indices = rng.integers(
            0,
            len(patient_df),
            size=len(patient_df),
        )

        labels = labels_all[indices]
        scores = scores_all[indices]

        if (
            metric in {"roc_auc", "auprc"}
            and np.unique(labels).size < 2
        ):
            continue

        predictions = (
            scores >= float(threshold)
        ).astype(int)

        if metric == "roc_auc":
            value = roc_auc_score(
                labels,
                scores,
            )
        elif metric == "auprc":
            value = average_precision_score(
                labels,
                scores,
            )
        elif metric == "accuracy":
            value = accuracy_score(
                labels,
                predictions,
            )
        elif metric == "balanced_accuracy":
            value = balanced_accuracy_score(
                labels,
                predictions,
            )
        elif metric == "sensitivity":
            value = recall_score(
                labels,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        elif metric == "specificity":
            value = specificity_score(
                labels,
                predictions,
            )
        elif metric == "f1":
            value = f1_score(
                labels,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        elif metric == "mcc":
            value = matthews_corrcoef(
                labels,
                predictions,
            )
        else:
            raise ValueError(
                f"Unknown bootstrap metric: {metric}"
            )

        values.append(float(value))

    if not values:
        return [
            float("nan"),
            float("nan"),
        ]

    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def cluster_bootstrap_image_metric(
    image_df: pd.DataFrame,
    metric: str,
    threshold: float = 0.5,
    value_column: str | None = None,
    samples: int = 2000,
    seed: int = SEED,
):
    patient_ids = (
        image_df["patient_id"]
        .drop_duplicates()
        .tolist()
    )

    grouped = {
        patient_id: group.reset_index(
            drop=True
        )
        for patient_id, group
        in image_df.groupby("patient_id")
    }

    rng = np.random.default_rng(seed)
    values = []

    for _ in range(int(samples)):
        sampled_patients = rng.choice(
            patient_ids,
            size=len(patient_ids),
            replace=True,
        )

        pieces = []

        for bootstrap_index, patient_id in enumerate(
            sampled_patients
        ):
            piece = grouped[patient_id].copy()
            piece["_bootstrap_patient"] = (
                bootstrap_index
            )
            pieces.append(piece)

        boot = pd.concat(
            pieces,
            ignore_index=True,
        )

        if value_column is not None:
            value = float(
                pd.to_numeric(
                    boot[value_column],
                    errors="coerce",
                )
                .dropna()
                .mean()
            )
            values.append(value)
            continue

        labels = (
            boot["label"].to_numpy(
                dtype=int
            )
        )
        scores = (
            boot[
                "probability_malignant"
            ].to_numpy(dtype=float)
        )

        if (
            metric in {"roc_auc", "auprc"}
            and np.unique(labels).size < 2
        ):
            continue

        predictions = (
            scores >= float(threshold)
        ).astype(int)

        if metric == "roc_auc":
            value = roc_auc_score(
                labels,
                scores,
            )
        elif metric == "auprc":
            value = average_precision_score(
                labels,
                scores,
            )
        elif metric == "balanced_accuracy":
            value = balanced_accuracy_score(
                labels,
                predictions,
            )
        else:
            raise ValueError(
                f"Unknown image bootstrap metric: {metric}"
            )

        values.append(float(value))

    if not values:
        return [
            float("nan"),
            float("nan"),
        ]

    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


"""
Canonical FINAL ConvNeXt-Tiny evaluation for the reviewed ThyroidXL notebook.

This is the publication-performance evaluator.

It intentionally does NOT evaluate the final 9,541-trained network on Fold 1,
because Fold 1 is part of the final network's training data.

Flow
----
1. Verify exact final checkpoint + matching training manifest.
2. Verify the checkpoint was refit on all 9,541 / 3,354 official-training data.
3. Only then open official test metadata.
4. Verify 2,094 test images / 739 test patients and zero train/test
   patient + filename overlap.
5. Run inference exactly once with frozen preprocessing and reporting rules.
6. Report:
   - image ROC-AUC/AUPRC
   - patient ROC-AUC/AUPRC using mean frame probability
   - fixed 0.5 operating point (PRIMARY)
   - development Youden threshold (SECONDARY)
   - confidence-weighted majority vote (SECONDARY benchmark-compatible)
   - sensitivity/specificity/accuracy/balanced accuracy/precision/F1/MCC
   - Brier/ECE calibration
   - auxiliary segmentation Dice/IoU
   - patient-level and patient-cluster bootstrap 95% CIs

No threshold or model choice is optimized using the official test.
"""

BATCH_SIZE = 16
NUM_WORKERS = 0
BOOTSTRAP_SAMPLES = 2000

OUT = (
    ROOT
    / "results"
    / "ConvNeXtTiny"
    / "FinalOfficialTest"
)
OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def run_final_inference(
    model,
    checkpoint,
    test_frame,
    device,
):
    dataset = ThyroidXLImageMaskDataset(
        test_frame,
        split="test",
        image_size=int(
            checkpoint["image_size"]
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    rows = []

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc="ConvNeXt-Tiny official held-out test",
            unit="batch",
        ):
            images = batch["image"].to(
                device,
                non_blocking=True,
            )
            masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            logits, segmentation_logits = (
                model(images)
            )
            probabilities = torch.sigmoid(
                logits
            )

            # Secondary audit only: reproduce the historical 512x512 padded
            # metrics.  Primary publication segmentation metrics below are in
            # ORIGINAL ultrasound coordinates, matching the custom YOLO metrics.
            segmentation_padded = (
                hard_segmentation_metrics(
                    segmentation_logits,
                    masks,
                    threshold=0.5,
                )
            )

            segmentation_probabilities = torch.sigmoid(
                segmentation_logits
            ).detach().cpu().numpy()
            valid_regions = (
                batch["valid_region"]
                .detach()
                .cpu()
                .numpy()
            )

            for index in range(
                len(batch["filename"])
            ):
                probability = float(
                    probabilities[
                        index
                    ].item()
                )

                original_segmentation = (
                    original_geometry_segmentation_metrics(
                        segmentation_probabilities[index, 0],
                        valid_regions[index, 0],
                        batch["mask_path"][index],
                        int(batch["original_height"][index]),
                        int(batch["original_width"][index]),
                        threshold=0.5,
                    )
                )

                rows.append(
                    {
                        "filename": str(
                            batch["filename"][
                                index
                            ]
                        ),
                        "patient_id": str(
                            batch["patient_id"][
                                index
                            ]
                        ),
                        "label": int(
                            batch["label"][
                                index
                            ].item()
                        ),
                        "probability_malignant": (
                            probability
                        ),
                        "prediction_at_0_5": int(
                            probability >= 0.5
                        ),
                        # -------------------------------------------------
                        # PRIMARY segmentation: ORIGINAL ultrasound geometry
                        # -------------------------------------------------
                        "segmentation_dice": float(
                            original_segmentation["dice"]
                        ),
                        "segmentation_iou": float(
                            original_segmentation["iou"]
                        ),
                        "predicted_mask_fraction": float(
                            original_segmentation[
                                "predicted_fraction"
                            ]
                        ),
                        "expert_mask_fraction": float(
                            original_segmentation[
                                "expert_fraction"
                            ]
                        ),
                        "segmentation_geometry": (
                            "original_ultrasound"
                        ),
                        "original_height": int(
                            batch["original_height"][index]
                        ),
                        "original_width": int(
                            batch["original_width"][index]
                        ),
                        "valid_y0": int(
                            original_segmentation["valid_y0"]
                        ),
                        "valid_y1": int(
                            original_segmentation["valid_y1"]
                        ),
                        "valid_x0": int(
                            original_segmentation["valid_x0"]
                        ),
                        "valid_x1": int(
                            original_segmentation["valid_x1"]
                        ),
                        # -------------------------------------------------
                        # SECONDARY audit: historical padded 512x512 metrics
                        # -------------------------------------------------
                        "segmentation_dice_padded_512": float(
                            segmentation_padded["dice"][
                                index
                            ].item()
                        ),
                        "segmentation_iou_padded_512": float(
                            segmentation_padded["iou"][
                                index
                            ].item()
                        ),
                        "predicted_mask_fraction_padded_512": float(
                            segmentation_padded[
                                "predicted_fraction"
                            ][index].item()
                        ),
                        "expert_mask_fraction_padded_512": float(
                            segmentation_padded[
                                "expert_fraction"
                            ][index].item()
                        ),
                    }
                )

    result = pd.DataFrame(rows)

    if len(result) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            "Final inference row count mismatch."
        )

    if (
        result["patient_id"].nunique()
        != EXPECTED_TEST_PATIENTS
    ):
        raise RuntimeError(
            "Final inference patient count mismatch."
        )

    return result


def save_figures(
    image_df,
    patient_df,
):
    from sklearn.metrics import (
        RocCurveDisplay,
        PrecisionRecallDisplay,
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )
    RocCurveDisplay.from_predictions(
        patient_df["label"],
        patient_df[
            "probability_malignant"
        ],
        ax=ax,
        name="ConvNeXt-Tiny",
    )
    ax.set_title(
        "ThyroidXL held-out patient ROC"
    )
    fig.tight_layout()
    fig.savefig(
        OUT / "patient_roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )
    PrecisionRecallDisplay.from_predictions(
        patient_df["label"],
        patient_df[
            "probability_malignant"
        ],
        ax=ax,
        name="ConvNeXt-Tiny",
    )
    ax.set_title(
        "ThyroidXL held-out patient precision-recall"
    )
    fig.tight_layout()
    fig.savefig(
        OUT / "patient_precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    probabilities = (
        patient_df[
            "probability_malignant"
        ].to_numpy(dtype=float)
    )
    labels = (
        patient_df["label"]
        .to_numpy(dtype=int)
    )

    bins = np.linspace(
        0.0,
        1.0,
        11,
    )
    xs = []
    ys = []

    for index in range(10):
        if index == 9:
            mask = (
                (probabilities >= bins[index])
                & (
                    probabilities
                    <= bins[index + 1]
                )
            )
        else:
            mask = (
                (probabilities >= bins[index])
                & (
                    probabilities
                    < bins[index + 1]
                )
            )

        if mask.any():
            xs.append(
                float(
                    probabilities[
                        mask
                    ].mean()
                )
            )
            ys.append(
                float(
                    labels[mask].mean()
                )
            )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )
    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        label="Perfect calibration",
    )
    ax.plot(
        xs,
        ys,
        marker="o",
        label="ConvNeXt-Tiny",
    )
    ax.set_xlabel(
        "Predicted malignant probability"
    )
    ax.set_ylabel(
        "Observed malignant fraction"
    )
    ax.set_title(
        "ThyroidXL held-out patient calibration"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUT / "patient_calibration.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    matrix = confusion_matrix(
        patient_df["label"],
        patient_df[
            "prediction_at_0_5"
        ],
        labels=[0, 1],
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )
    shown = ax.imshow(matrix)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(
        ["Benign", "Malignant"]
    )
    ax.set_yticklabels(
        ["Benign", "Malignant"]
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        "Patient confusion matrix @ 0.5"
    )

    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                str(
                    int(
                        matrix[
                            row,
                            column,
                        ]
                    )
                ),
                ha="center",
                va="center",
            )

    fig.colorbar(
        shown,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(
        OUT
        / "patient_confusion_matrix_0_5.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )
    ax.hist(
        image_df[
            "segmentation_dice"
        ],
        bins=30,
    )
    ax.set_xlabel(
        "Per-image Dice"
    )
    ax.set_ylabel("Images")
    ax.set_title(
        "ConvNeXt-Tiny auxiliary segmentation — held-out Dice (original geometry)"
    )
    fig.tight_layout()
    fig.savefig(
        OUT
        / "segmentation_dice_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    (
        checkpoint_path,
        checkpoint,
        manifest_path,
        manifest,
    ) = find_final_checkpoint_and_manifest()

    checkpoint_sha = sha256_file(
        checkpoint_path
    )

    print("=" * 80)
    print(
        "THYROIDXL MOBILENETV3 — FINAL PUBLICATION EVALUATION"
    )
    print("=" * 80)
    print(
        "Checkpoint:",
        checkpoint_path,
    )
    print(
        "Checkpoint SHA256:",
        checkpoint_sha,
    )
    print(
        "Training manifest:",
        manifest_path,
    )
    print(
        "Final training cohort:",
        checkpoint["training_images"],
        "images /",
        checkpoint["training_patients"],
        "patients",
    )
    print(
        "Official test opened so far: NO"
    )
    print()

    # Training metadata can be opened freely. The official test is opened
    # only after the final checkpoint/manifest have passed all checks.
    official_train = (
        load_official_train_metadata()
    )

    test_frame = (
        load_official_test_metadata_after_freeze(
            official_train
        )
    )

    print(
        "Official held-out cohort:",
        len(test_frame),
        "images /",
        test_frame[
            "patient_id"
        ].nunique(),
        "patients",
    )
    print(
        "Train/test patient overlap: 0"
    )
    print(
        "Train/test filename overlap: 0"
    )
    print(
        "No test-set tuning is performed."
    )
    print(
        "Segmentation metrics: ORIGINAL ultrasound coordinates "
        "after inverse-mapping the ConvNeXt-Tiny probability map."
    )
    print()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = load_model(
        checkpoint_path,
        checkpoint,
        device,
    )

    image_df = run_final_inference(
        model,
        checkpoint,
        test_frame,
        device,
    )

    patient_df = aggregate_patient_mean(
        image_df
    )
    wmv_df = aggregate_patient_wmv(
        image_df,
        image_threshold=(
            PRIMARY_IMAGE_THRESHOLD
        ),
    )

    development_threshold = float(
        checkpoint[
            "development_selected_patient_threshold"
        ]
    )

    patient_df[
        "prediction_at_0_5"
    ] = (
        patient_df[
            "probability_malignant"
        ]
        >= PRIMARY_PATIENT_THRESHOLD
    ).astype(int)

    patient_df[
        "prediction_at_development_threshold"
    ] = (
        patient_df[
            "probability_malignant"
        ]
        >= development_threshold
    ).astype(int)

    # Canonical publication-schema aliases used by downstream paired analyses.
    # These do not change any operating rule: ConvNeXt-Tiny primary remains fixed 0.5.
    image_df["prediction_primary"] = image_df["prediction_at_0_5"].astype(int)
    image_df["correct_primary"] = (
        image_df["prediction_primary"] == image_df["label"]
    ).astype(int)
    patient_df["prediction_primary"] = patient_df["prediction_at_0_5"].astype(int)
    patient_df["correct_primary"] = (
        patient_df["prediction_primary"] == patient_df["label"]
    ).astype(int)

    image_metrics_05 = (
        classification_metrics(
            image_df["label"],
            image_df[
                "probability_malignant"
            ],
            PRIMARY_IMAGE_THRESHOLD,
        )
    )

    patient_metrics_05 = (
        classification_metrics(
            patient_df["label"],
            patient_df[
                "probability_malignant"
            ],
            PRIMARY_PATIENT_THRESHOLD,
        )
    )

    patient_metrics_development = (
        classification_metrics(
            patient_df["label"],
            patient_df[
                "probability_malignant"
            ],
            development_threshold,
        )
    )

    patient_metrics_wmv = (
        hard_prediction_metrics(
            wmv_df["label"],
            wmv_df["prediction_wmv"],
        )
    )

    bootstrap = {
        "patient_roc_auc": (
            patient_bootstrap_ci(
                patient_df,
                "roc_auc",
                PRIMARY_PATIENT_THRESHOLD,
                BOOTSTRAP_SAMPLES,
            )
        ),
        "patient_auprc": (
            patient_bootstrap_ci(
                patient_df,
                "auprc",
                PRIMARY_PATIENT_THRESHOLD,
                BOOTSTRAP_SAMPLES,
            )
        ),
        "patient_accuracy_0_5": (
            patient_bootstrap_ci(
                patient_df,
                "accuracy",
                PRIMARY_PATIENT_THRESHOLD,
                BOOTSTRAP_SAMPLES,
            )
        ),
        "patient_balanced_accuracy_0_5": (
            patient_bootstrap_ci(
                patient_df,
                "balanced_accuracy",
                PRIMARY_PATIENT_THRESHOLD,
                BOOTSTRAP_SAMPLES,
            )
        ),
        "patient_sensitivity_0_5": (
            patient_bootstrap_ci(
                patient_df,
                "sensitivity",
                PRIMARY_PATIENT_THRESHOLD,
                BOOTSTRAP_SAMPLES,
            )
        ),
        "patient_specificity_0_5": (
            patient_bootstrap_ci(
                patient_df,
                "specificity",
                PRIMARY_PATIENT_THRESHOLD,
                BOOTSTRAP_SAMPLES,
            )
        ),
        "patient_f1_0_5": (
            patient_bootstrap_ci(
                patient_df,
                "f1",
                PRIMARY_PATIENT_THRESHOLD,
                BOOTSTRAP_SAMPLES,
            )
        ),
        "patient_mcc_0_5": (
            patient_bootstrap_ci(
                patient_df,
                "mcc",
                PRIMARY_PATIENT_THRESHOLD,
                BOOTSTRAP_SAMPLES,
            )
        ),
        "image_roc_auc_patient_cluster": (
            cluster_bootstrap_image_metric(
                image_df,
                metric="roc_auc",
                threshold=(
                    PRIMARY_IMAGE_THRESHOLD
                ),
                samples=(
                    BOOTSTRAP_SAMPLES
                ),
            )
        ),
        "image_auprc_patient_cluster": (
            cluster_bootstrap_image_metric(
                image_df,
                metric="auprc",
                threshold=(
                    PRIMARY_IMAGE_THRESHOLD
                ),
                samples=(
                    BOOTSTRAP_SAMPLES
                ),
            )
        ),
        "mean_segmentation_dice_patient_cluster": (
            cluster_bootstrap_image_metric(
                image_df,
                metric="mean",
                value_column=(
                    "segmentation_dice"
                ),
                samples=(
                    BOOTSTRAP_SAMPLES
                ),
            )
        ),
        "mean_segmentation_iou_patient_cluster": (
            cluster_bootstrap_image_metric(
                image_df,
                metric="mean",
                value_column=(
                    "segmentation_iou"
                ),
                samples=(
                    BOOTSTRAP_SAMPLES
                ),
            )
        ),
        "mean_segmentation_dice_padded_512_patient_cluster_secondary": (
            cluster_bootstrap_image_metric(
                image_df,
                metric="mean",
                value_column=(
                    "segmentation_dice_padded_512"
                ),
                samples=(
                    BOOTSTRAP_SAMPLES
                ),
                seed=SEED + 100,
            )
        ),
        "mean_segmentation_iou_padded_512_patient_cluster_secondary": (
            cluster_bootstrap_image_metric(
                image_df,
                metric="mean",
                value_column=(
                    "segmentation_iou_padded_512"
                ),
                samples=(
                    BOOTSTRAP_SAMPLES
                ),
                seed=SEED + 200,
            )
        ),
    }

    summary = {
        "status": (
            "THYROIDXL_CONVNEXTTINY_FINAL_PUBLICATION_EVALUATION"
        ),
        "dataset": "ThyroidXL",
        "dataset_repo": REPO_ID,
        "dataset_revision": REPO_REVISION,
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_sha256": (
            checkpoint_sha
        ),
        "training_manifest": str(
            manifest_path
        ),
        "official_training_images": (
            EXPECTED_TRAIN_IMAGES
        ),
        "official_training_patients": (
            EXPECTED_TRAIN_PATIENTS
        ),
        "official_test_images": (
            EXPECTED_TEST_IMAGES
        ),
        "official_test_patients": (
            EXPECTED_TEST_PATIENTS
        ),
        "train_test_patient_overlap": 0,
        "train_test_filename_overlap": 0,
        "official_test_used_for_tuning": (
            False
        ),
        "reporting_policy": {
            "primary_discrimination": [
                "ROC-AUC",
                "AUPRC",
            ],
            "primary_image_threshold": (
                PRIMARY_IMAGE_THRESHOLD
            ),
            "primary_patient_threshold": (
                PRIMARY_PATIENT_THRESHOLD
            ),
            "secondary_patient_threshold": (
                development_threshold
            ),
            "secondary_patient_threshold_source": (
                "Fold-1 development Youden J "
                "saved before final refit"
            ),
            "primary_patient_aggregation": (
                "mean malignant probability "
                "across all patient frames"
            ),
            "secondary_benchmark_patient_aggregation": (
                "confidence-weighted majority voting"
            ),
        },
        "image_metrics_0_5": (
            image_metrics_05
        ),
        "patient_metrics_0_5_primary": (
            patient_metrics_05
        ),
        "patient_metrics_development_threshold_secondary": (
            patient_metrics_development
        ),
        "patient_metrics_wmv_secondary": (
            patient_metrics_wmv
        ),
        "calibration": {
            "image_ece_10": (
                expected_calibration_error(
                    image_df["label"],
                    image_df[
                        "probability_malignant"
                    ],
                    10,
                )
            ),
            "patient_ece_10": (
                expected_calibration_error(
                    patient_df["label"],
                    patient_df[
                        "probability_malignant"
                    ],
                    10,
                )
            ),
            "image_brier": float(
                brier_score_loss(
                    image_df["label"],
                    image_df[
                        "probability_malignant"
                    ],
                )
            ),
            "patient_brier": float(
                brier_score_loss(
                    patient_df["label"],
                    patient_df[
                        "probability_malignant"
                    ],
                )
            ),
        },
        "segmentation": {
            "threshold": 0.5,
            "mean_dice": float(
                image_df[
                    "segmentation_dice"
                ].mean()
            ),
            "median_dice": float(
                image_df[
                    "segmentation_dice"
                ].median()
            ),
            "mean_iou": float(
                image_df[
                    "segmentation_iou"
                ].mean()
            ),
            "median_iou": float(
                image_df[
                    "segmentation_iou"
                ].median()
            ),
            "geometry": "original_ultrasound",
            "coordinate_policy": (
                "Generate the auxiliary segmentation probability map on the "
                "exact aspect-ratio-preserving padded 512x512 model input; "
                "after inference remove only synthetic preprocessing padding, "
                "resize the valid probability map to the original ultrasound "
                "dimensions, apply the frozen 0.5 mask threshold, and compare "
                "with the untouched original expert mask."
            ),
            "directly_comparable_to_custom_yolo_original_geometry": True,
            "historical_padded_512_secondary": {
                "mean_dice": float(
                    image_df[
                        "segmentation_dice_padded_512"
                    ].mean()
                ),
                "median_dice": float(
                    image_df[
                        "segmentation_dice_padded_512"
                    ].median()
                ),
                "mean_iou": float(
                    image_df[
                        "segmentation_iou_padded_512"
                    ].mean()
                ),
                "median_iou": float(
                    image_df[
                        "segmentation_iou_padded_512"
                    ].median()
                ),
                "geometry": (
                    "512x512 aspect-ratio-preserving resize/pad evaluation space"
                ),
            },
        },
        "bootstrap_95_ci": bootstrap,
    }

    image_df.to_csv(
        OUT
        / "test_image_predictions.csv",
        index=False,
    )
    patient_df.to_csv(
        OUT
        / "test_patient_predictions.csv",
        index=False,
    )
    wmv_df.to_csv(
        OUT
        / "test_patient_predictions_wmv.csv",
        index=False,
    )

    (
        OUT / "test_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_figures(
        image_df,
        patient_df,
    )

    print("=" * 80)
    print(
        "PRIMARY PATIENT METRICS @ 0.5"
    )
    print("=" * 80)
    print(
        json.dumps(
            patient_metrics_05,
            indent=2,
        )
    )

    print()
    print("=" * 80)
    print(
        "SECONDARY PATIENT METRICS "
        "@ DEVELOPMENT THRESHOLD"
    )
    print("=" * 80)
    print(
        json.dumps(
            patient_metrics_development,
            indent=2,
        )
    )

    print()
    print("=" * 80)
    print(
        "SECONDARY THYROIDXL-COMPATIBLE WMV"
    )
    print("=" * 80)
    print(
        json.dumps(
            patient_metrics_wmv,
            indent=2,
        )
    )

    print()
    print(
        "Saved:",
        OUT,
    )
    print(
        "FINAL TEST COMPLETE. No test-set "
        "threshold/model/protocol tuning occurred."
    )


# =============================================================================
# MATCHED XAI EVALUATION — ORIGINAL ULTRASOUND COORDINATES
# =============================================================================
# This section is intentionally independent of the classification/segmentation
# publication evaluator above. It reuses only the exact model reconstruction,
# checkpoint verification, dataset metadata, and frozen evaluation preprocessing.
#
# Matched protocol across all CNNs:
#   - ONE final publication checkpoint only.
#   - same architecture-only layer rule: deepest executed backbone Conv2d at 16x16.
#   - Grad-CAM, Grad-CAM++, and Layer-CAM on that same layer.
#   - target = model-predicted binary class at the fixed 0.5 reference threshold.
#   - CAM is generated on the exact padded 512x512 model input.
#   - synthetic padding is removed only AFTER CAM generation.
#   - localisation metrics are computed in the ORIGINAL ultrasound coordinates.
#   - expert/YOLO masks are NEVER used to create, crop, select, or tune a CAM.
#   - no held-out XAI outcome is used to select a model, method, layer, threshold,
#     smoothing option, or preprocessing rule.
# =============================================================================

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
except ImportError as exc:
    raise ImportError(
        "Missing XAI dependency. Install it with:\n"
        "  pip install grad-cam"
    ) from exc

MODEL_DISPLAY_NAME = 'ConvNeXt-Tiny'
MODEL_RESULT_DIR = 'ConvNeXtTiny'
MODEL_XAI_TAG = 'CONVNEXTTINY'

TARGET_RESOLUTION = 16
TOP_FRACTION = 0.15
HIT_THRESHOLD = 0.50
XAI_BOOTSTRAP_SAMPLES = 2000
XAI_MAX_CASES = 0  # 0 = complete official held-out cohort

METHODS = {
    "Grad-CAM": GradCAM,
    "Grad-CAM++": GradCAMPlusPlus,
    "Layer-CAM": LayerCAM,
}

METHOD_TARGET_RESOLUTIONS = {
    'Grad-CAM': [16],
    'Grad-CAM++': [16],
    'Layer-CAM': [16],
}

XAI_OUT = (
    ROOT
    / "results"
    / MODEL_RESULT_DIR
    / "XAI_AnatomicalCoordinates"
)
XAI_OUT.mkdir(parents=True, exist_ok=True)


class ClassificationOnlyWrapper(nn.Module):
    """Expose only the frozen binary classification logit to CAM methods."""

    def __init__(self, trained_model):
        super().__init__()
        self.trained_model = trained_model

    def forward(self, image):
        logits = self.trained_model.classify(image)
        return logits.reshape(-1, 1)


class PredictedBinaryClassTarget:
    """
    Single-logit target used identically for every CNN.

    probability >= 0.5 -> explain +malignant logit
    probability <  0.5 -> explain -malignant logit
    """

    def __init__(self, predicted_class: int):
        self.sign = 1.0 if int(predicted_class) == 1 else -1.0

    def __call__(self, model_output):
        return self.sign * model_output.reshape(-1)[0]


def module_by_name(model, name: str):
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Module not found: {name}")
    return modules[name]


def normalize_cam(cam):
    cam = np.asarray(cam, dtype=np.float32)
    low = float(cam.min())
    high = float(cam.max())
    if high <= low:
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - low) / (high - low)).astype(np.float32)


def top_fraction_mask(cam, fraction=TOP_FRACTION):
    flat = np.asarray(cam, dtype=np.float32).reshape(-1)
    active = max(1, int(math.ceil(flat.size * float(fraction))))
    indices = np.argpartition(flat, -active)[-active:]
    output = np.zeros(flat.size, dtype=np.uint8)
    output[indices] = 1
    return output.reshape(cam.shape)


def localisation_metrics(cam, expert_mask):
    """Metrics are calculated only in the original ultrasound geometry."""
    cam = normalize_cam(cam)
    ground_truth = np.asarray(expert_mask) > 0
    selected = top_fraction_mask(cam) > 0

    intersection = float(np.logical_and(selected, ground_truth).sum())
    union = float(np.logical_or(selected, ground_truth).sum())
    selected_sum = float(selected.sum())
    gt_sum = float(ground_truth.sum())

    iou = intersection / union if union else 0.0
    dice = (
        2.0 * intersection / (selected_sum + gt_sum)
        if (selected_sum + gt_sum)
        else 0.0
    )

    # BTXRD-style thresholded lesion-overlap hit:
    # at least one expert-mask pixel reaches normalized CAM >= 0.5.
    overlap_hit = float(
        np.logical_and(ground_truth, cam >= HIT_THRESHOLD).any()
    )

    # Strict pointing game: the single hottest CAM pixel is inside the nodule.
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    pointing_hit = float(ground_truth[peak])

    nonnegative = np.clip(cam, 0.0, None)
    total_energy = float(nonnegative.sum())
    nodule_energy = (
        float(nonnegative[ground_truth].sum() / total_energy)
        if total_energy > 0.0
        else 0.0
    )

    inside = float(cam[ground_truth].mean()) if ground_truth.any() else 0.0
    outside = float(cam[~ground_truth].mean()) if (~ground_truth).any() else 0.0

    return {
        "top15_iou": float(iou),
        "top15_dice": float(dice),
        "overlap_hit_0_5": float(overlap_hit),
        "strict_pointing_hit": float(pointing_hit),
        "nodule_energy_fraction": float(nodule_energy),
        "inside_mean_activation": inside,
        "outside_mean_activation": outside,
        "inside_outside_ratio": float(inside / (outside + 1e-8)),
    }


def make_xai_transform(image_size: int):
    """
    Exact evaluation image preprocessing plus an all-ones valid-region mask.

    The valid-region mask is NOT a lesion mask and is never shown to the model.
    It records which padded-canvas pixels originated from the real ultrasound.
    """
    return A.Compose(
        [
            A.LongestMaxSize(
                max_size=int(image_size),
                area_for_downscale="image",
            ),
            A.PadIfNeeded(
                min_height=int(image_size),
                min_width=int(image_size),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
            ),
            A.Normalize(
                mean=IMAGE_MEAN,
                std=IMAGE_STD,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        additional_targets={"valid_region": "mask"},
        seed=SEED,
        strict=True,
    )


class AnatomicalXAIDataset(Dataset):
    def __init__(self, frame, split, image_size, max_cases=0):
        frame = frame.reset_index(drop=True)
        if int(max_cases) > 0:
            frame = frame.iloc[: int(max_cases)].copy()
        self.frame = frame
        self.split = str(split)
        self.transform = make_xai_transform(int(image_size))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        filename = str(row["filename"])

        image_path = fetch(f"{self.split}/images/{filename}")
        mask_path = fetch(f"{self.split}/masks/{filename}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Could not read XAI image/mask pair: {filename}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask > 0).astype(np.uint8)

        if image.shape[:2] != mask.shape[:2]:
            raise RuntimeError(
                f"XAI image/mask shape mismatch for {filename}: "
                f"{image.shape[:2]} vs {mask.shape[:2]}"
            )
        if not mask.any():
            raise RuntimeError(f"Empty expert nodule mask for XAI: {filename}")

        original_height = int(image.shape[0])
        original_width = int(image.shape[1])
        valid_region = np.ones(
            (original_height, original_width),
            dtype=np.uint8,
        )

        transformed = self.transform(
            image=image,
            mask=mask,
            valid_region=valid_region,
        )

        image_tensor = transformed["image"].float()
        transformed_mask = np.asarray(transformed["mask"]) > 0
        transformed_valid = np.asarray(transformed["valid_region"]) > 0

        if transformed_mask.ndim == 3:
            transformed_mask = np.squeeze(transformed_mask)
        if transformed_valid.ndim == 3:
            transformed_valid = np.squeeze(transformed_valid)

        if transformed_valid.shape != (
            int(image_tensor.shape[-2]),
            int(image_tensor.shape[-1]),
        ):
            raise RuntimeError(
                f"Valid-region shape mismatch for {filename}: "
                f"{transformed_valid.shape} vs {tuple(image_tensor.shape[-2:])}"
            )
        if not transformed_valid.any():
            raise RuntimeError(f"Valid-region mask became empty for {filename}.")

        if np.logical_and(transformed_mask, ~transformed_valid).any():
            raise RuntimeError(
                "Transformed expert mask overlaps synthetic padding for "
                f"{filename}."
            )

        return {
            "image": image_tensor,
            "original_mask": mask,
            "valid_region": transformed_valid.astype(np.uint8),
            "original_height": original_height,
            "original_width": original_width,
            "filename": filename,
            "patient_id": str(row["patient_id"]),
            "label": int(row["label"]),
        }


def valid_region_bounds(valid_region):
    valid = np.asarray(valid_region) > 0
    if valid.ndim != 2 or not valid.any():
        raise ValueError("valid_region must be a non-empty 2D binary mask.")

    rows, cols = np.where(valid)
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(cols.min()), int(cols.max()) + 1

    rectangle = np.zeros_like(valid, dtype=bool)
    rectangle[y0:y1, x0:x1] = True
    if not np.array_equal(valid, rectangle):
        raise RuntimeError(
            "Valid-region geometry is not one contiguous rectangular image area."
        )
    return y0, y1, x0, x1


def inverse_map_cam_to_original(
    padded_cam,
    valid_region,
    original_height,
    original_width,
):
    """
    Remove ONLY synthetic preprocessing padding after CAM generation, then
    resize the valid CAM rectangle to the untouched original image geometry.
    """
    cam = np.asarray(padded_cam, dtype=np.float32)
    valid = np.asarray(valid_region) > 0
    if cam.shape != valid.shape:
        raise RuntimeError(
            f"CAM/valid-region shape mismatch: cam={cam.shape}, valid={valid.shape}"
        )

    y0, y1, x0, x1 = valid_region_bounds(valid)
    cropped = cam[y0:y1, x0:x1]
    if cropped.size == 0:
        raise RuntimeError("CAM crop after removing padding is empty.")

    anatomical_cam = cv2.resize(
        cropped,
        (int(original_width), int(original_height)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    expected_shape = (int(original_height), int(original_width))
    if anatomical_cam.shape != expected_shape:
        raise RuntimeError(
            f"Inverse-mapped CAM has wrong shape: "
            f"{anatomical_cam.shape} vs {expected_shape}"
        )
    return anatomical_cam, (y0, y1, x0, x1)


def padding_audit_metrics(padded_cam, valid_region):
    """Audit attribution assigned to synthetic preprocessing padding."""
    cam = normalize_cam(padded_cam)
    valid = np.asarray(valid_region) > 0
    if cam.shape != valid.shape:
        raise RuntimeError("CAM/valid-region shape mismatch in padding audit.")

    nonnegative = np.clip(cam, 0.0, None)
    total = float(nonnegative.sum())
    padding = ~valid
    padding_energy_fraction = (
        float(nonnegative[padding].sum() / total)
        if total > 0.0 and padding.any()
        else 0.0
    )
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)

    return {
        "padding_energy_fraction": float(padding_energy_fraction),
        "peak_in_padding": int(not bool(valid[peak])),
        "valid_canvas_fraction": float(valid.mean()),
    }


def fixed_backbone_target_layer(
    model,
    image_size,
    target_resolution=TARGET_RESOLUTION,
):
    """
    Architecture-only target-layer rule used identically for every CNN.

    Choose the deepest EXECUTED backbone Conv2d whose feature map is exactly
    16x16 for the frozen 512x512 input. Discovery uses a zero-valued dummy
    tensor only — no ThyroidXL image, label, mask, validation score, or test
    result can influence this choice.
    """
    records = []
    handles = []

    def hook_factory(name):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                records.append(
                    (name, int(output.shape[-2]), int(output.shape[-1]))
                )
        return hook

    for name, module in model.named_modules():
        if name.startswith("backbone.") and isinstance(module, nn.Conv2d):
            handles.append(
                module.register_forward_hook(hook_factory(name))
            )

    device = next(model.parameters()).device
    dummy = torch.zeros(
        1,
        3,
        int(image_size),
        int(image_size),
        device=device,
    )

    model.eval()
    with torch.no_grad():
        _ = model.classify(dummy)

    for handle in handles:
        handle.remove()

    candidates = [
        name
        for name, height, width in records
        if height == int(target_resolution)
        and width == int(target_resolution)
    ]
    if not candidates:
        observed = sorted(set((h, w) for _, h, w in records))
        raise RuntimeError(
            "No executed backbone Conv2d was found at "
            f"{target_resolution}x{target_resolution}. Observed={observed}"
        )

    # records are in execution order: the final candidate is the deepest one.
    return candidates[-1]


def evaluate_method_layer(
    trained_model,
    frame,
    method_name,
    layer_names,
    target_resolutions,
    progress_bar,
    max_cases=0,
):
    device = next(trained_model.parameters()).device
    wrapper = ClassificationOnlyWrapper(trained_model).to(device)
    wrapper.eval()

    layer_names = [str(name) for name in layer_names]
    target_resolutions = [int(value) for value in target_resolutions]
    if len(layer_names) != len(target_resolutions):
        raise RuntimeError(
            f"{method_name}: target layer/resolution count mismatch."
        )

    target_layers = [
        module_by_name(trained_model, name)
        for name in layer_names
    ]

    dataset = AnatomicalXAIDataset(
        frame,
        split="test",
        image_size=trained_model.image_size,
        max_cases=max_cases,
    )

    cam_class = METHODS[method_name]
    rows = []

    progress_bar.set_description(f"{MODEL_DISPLAY_NAME} XAI | {method_name}")

    with cam_class(
        model=wrapper,
        target_layers=target_layers,
    ) as cam_engine:
        for item in dataset:
            image = item["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                probability = float(
                    torch.sigmoid(trained_model.classify(image))[0].item()
                )

            predicted_class = int(probability >= 0.5)

            grayscale_cam = cam_engine(
                input_tensor=image,
                targets=[PredictedBinaryClassTarget(predicted_class)],
                aug_smooth=False,
                eigen_smooth=False,
            )[0]

            anatomical_cam, bounds = inverse_map_cam_to_original(
                grayscale_cam,
                item["valid_region"],
                item["original_height"],
                item["original_width"],
            )

            metrics = localisation_metrics(
                anatomical_cam,
                item["original_mask"],
            )
            padding_audit = padding_audit_metrics(
                grayscale_cam,
                item["valid_region"],
            )

            y0, y1, x0, x1 = bounds
            rows.append(
                {
                    "filename": item["filename"],
                    "patient_id": item["patient_id"],
                    "label": int(item["label"]),
                    "probability_malignant": probability,
                    "predicted_class": predicted_class,
                    "classification_correct": int(
                        predicted_class == int(item["label"])
                    ),
                    "method": method_name,
                    "target_layers": "|".join(layer_names),
                    "target_resolutions": "|".join(
                        str(value) for value in target_resolutions
                    ),
                    "target_layer": (
                        layer_names[0]
                        if len(layer_names) == 1
                        else "|".join(layer_names)
                    ),
                    "target_resolution": (
                        int(target_resolutions[0])
                        if len(target_resolutions) == 1
                        else "|".join(
                            str(value) for value in target_resolutions
                        )
                    ),
                    "n_target_layers": int(len(layer_names)),
                    "coordinate_space": "original_ultrasound",
                    "original_height": int(item["original_height"]),
                    "original_width": int(item["original_width"]),
                    "valid_y0": int(y0),
                    "valid_y1": int(y1),
                    "valid_x0": int(x0),
                    "valid_x1": int(x1),
                    **padding_audit,
                    **metrics,
                }
            )
            progress_bar.update(1)

    result = pd.DataFrame(rows)
    expected_images = int(max_cases) if int(max_cases) > 0 else len(frame)
    if len(result) != expected_images:
        raise RuntimeError(
            f"XAI inference row count mismatch for {method_name}: "
            f"expected {expected_images}, got {len(result)}."
        )
    return result


XAI_METRIC_COLUMNS = [
    "top15_iou",
    "top15_dice",
    "overlap_hit_0_5",
    "strict_pointing_hit",
    "nodule_energy_fraction",
    "inside_outside_ratio",
    "padding_energy_fraction",
    "peak_in_padding",
]


def patient_balanced_summary(frame):
    patient = frame.groupby("patient_id")[XAI_METRIC_COLUMNS].mean()
    return {
        "patient_balanced_mean_" + column: float(patient[column].mean())
        for column in XAI_METRIC_COLUMNS
    }


def patient_balanced_bootstrap_ci(
    frame,
    samples=XAI_BOOTSTRAP_SAMPLES,
    seed=SEED,
):
    """
    Patient bootstrap for patient-balanced XAI means.

    Each patient's frame-level metric is averaged first. Patients are then
    sampled with replacement, preserving the patient as the resampling unit.
    """
    patient = (
        frame.groupby("patient_id")[XAI_METRIC_COLUMNS]
        .mean()
        .sort_index()
    )
    values = patient.to_numpy(dtype=np.float64)
    n_patients = values.shape[0]
    rng = np.random.default_rng(int(seed))
    boot = np.empty((int(samples), values.shape[1]), dtype=np.float64)

    for index in range(int(samples)):
        draw = rng.integers(0, n_patients, size=n_patients)
        boot[index] = values[draw].mean(axis=0)

    low = np.quantile(boot, 0.025, axis=0)
    high = np.quantile(boot, 0.975, axis=0)

    output = {}
    for idx, column in enumerate(XAI_METRIC_COLUMNS):
        output[f"patient_balanced_{column}_ci95_low"] = float(low[idx])
        output[f"patient_balanced_{column}_ci95_high"] = float(high[idx])
    return output


def xai_main():
    # ------------------------------------------------------------------
    # 1. Verify the exact final publication checkpoint and manifest.
    # ------------------------------------------------------------------
    (
        final_path,
        final_checkpoint,
        manifest_path,
        manifest,
    ) = find_final_checkpoint_and_manifest()
    checkpoint_sha = sha256_file(final_path)

    print("=" * 80)
    print(f"{MODEL_DISPLAY_NAME.upper()} XAI — MATCHED ANATOMICAL-COORDINATE PROTOCOL")
    print("=" * 80)
    print("Final checkpoint:", final_path)
    print("Checkpoint SHA256:", checkpoint_sha)
    print("Training manifest:", manifest_path)
    print(
        "Final training cohort:",
        f"{EXPECTED_TRAIN_IMAGES} images / {EXPECTED_TRAIN_PATIENTS} patients",
    )
    print("Development checkpoint used: NO")
    print("Held-out XAI used to choose layer/method: NO")
    print()

    official_train = load_official_train_metadata()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    final_model = load_model(final_path, final_checkpoint, device)

    # ------------------------------------------------------------------
    # 2. Architecture-only XAI freeze.
    # ------------------------------------------------------------------
    chosen = {}

    for method_name in METHODS:
        resolutions = list(
            METHOD_TARGET_RESOLUTIONS[method_name]
        )
        layer_names = [
            fixed_backbone_target_layer(
                final_model,
                final_model.image_size,
                int(resolution),
            )
            for resolution in resolutions
        ]
        chosen[method_name] = {
            "target_layers": layer_names,
            "target_resolutions": resolutions,
            "n_target_layers": len(layer_names),
        }

    freeze = {
        "status": f"THYROIDXL_{MODEL_XAI_TAG}_XAI_MATCHED_ANATOMICAL_PROTOCOL_FROZEN",
        "protocol_version": "method_specific_anatomical_coordinates_v4",
        "dataset": "ThyroidXL",
        "dataset_repo": REPO_ID,
        "dataset_revision": REPO_REVISION,
        "model": MODEL_DISPLAY_NAME,
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": checkpoint_sha,
        "training_manifest": str(manifest_path),
        "final_training_images": int(EXPECTED_TRAIN_IMAGES),
        "final_training_patients": int(EXPECTED_TRAIN_PATIENTS),
        "development_checkpoint_used": False,
        "configuration_source": "matched_architecture_fixed_16x16_configuration",
        "xai_layer_selection_data_used": False,
        "official_test_used_for_xai_layer_selection": False,
        "selected_methods": chosen,
        "target_layer_policy": (
            "Method-specific locked target resolutions. Each requested "
            "resolution resolves to the deepest executed backbone Conv2d at "
            "that spatial resolution using architecture plus a zero-valued "
            "dummy tensor only."
        ),
        "same_target_layer_for_all_methods": False,
        "methods": list(METHODS.keys()),
        "selected_methods": chosen,
        "target_policy": (
            "predicted binary class at fixed 0.5 reference threshold: "
            "malignant -> +logit; benign -> -logit"
        ),
        "top_fraction": float(TOP_FRACTION),
        "hit_threshold": float(HIT_THRESHOLD),
        "coordinate_policy": (
            "Generate CAM on the exact padded model input; identify the real-image "
            "rectangle with an all-ones valid-region mask transformed by the same "
            "resize/pad operations; remove only synthetic padding after CAM "
            "generation; resize the valid CAM to original ultrasound geometry; "
            "compute localisation against the untouched original expert mask."
        ),
        "expert_mask_used_to_modify_cam": False,
        "yolo_mask_used_to_modify_cam": False,
        "cam_retuned_from_heldout_results": False,
        "bootstrap_unit": "patient",
        "bootstrap_samples": int(XAI_BOOTSTRAP_SAMPLES),
    }

    freeze_path = XAI_OUT / (
        MODEL_RESULT_DIR.lower() + "_xai_anatomical_freeze.json"
    )
    freeze_path.write_text(
        json.dumps(freeze, indent=2),
        encoding="utf-8",
    )

    print("Locked method-specific XAI configuration:")
    for method_name, config in chosen.items():
        print(
            f"  {method_name}: "
            f"resolutions={config['target_resolutions']} | "
            f"layers={config['target_layers']}"
        )
    print("Freeze:", freeze_path)
    print()

    # ------------------------------------------------------------------
    # 3. Full official held-out cohort, zero overlap verification.
    # ------------------------------------------------------------------
    test_frame = load_official_test_metadata_after_freeze(official_train)

    print("=" * 80)
    print(f"{MODEL_DISPLAY_NAME.upper()} XAI — OFFICIAL HELD-OUT TEST")
    print("=" * 80)
    print("Held-out images:", len(test_frame))
    print("Held-out patients:", test_frame["patient_id"].nunique())
    print("Train/test patient overlap: 0")
    print("Train/test filename overlap: 0")
    print("Localisation coordinate space: ORIGINAL ULTRASOUND")
    print("No held-out XAI tuning is performed.")
    print()

    total_cases = (
        min(int(XAI_MAX_CASES), len(test_frame))
        if int(XAI_MAX_CASES) > 0
        else len(test_frame)
    )
    total_cam_evaluations = total_cases * len(METHODS)
    print(
        "XAI workload:",
        f"{total_cases} images x {len(METHODS)} methods = "
        f"{total_cam_evaluations} CAM evaluations",
    )

    test_case_tables = []
    test_summary_rows = []

    with tqdm(
        total=total_cam_evaluations,
        desc=f"{MODEL_DISPLAY_NAME} XAI",
        unit="CAM",
        dynamic_ncols=True,
    ) as progress_bar:
        for method_index, method_name in enumerate(METHODS):
            config = chosen[method_name]
            result = evaluate_method_layer(
                final_model,
                test_frame,
                method_name=method_name,
                layer_names=config["target_layers"],
                target_resolutions=config["target_resolutions"],
                progress_bar=progress_bar,
                max_cases=XAI_MAX_CASES,
            )
            test_case_tables.append(result)

            overall = patient_balanced_summary(result)
            ci = patient_balanced_bootstrap_ci(
                result,
                samples=XAI_BOOTSTRAP_SAMPLES,
                seed=SEED + method_index,
            )

            correct = result[result["classification_correct"] == 1]
            incorrect = result[result["classification_correct"] == 0]
            correct_iou = (
                float(correct.groupby("patient_id")["top15_iou"].mean().mean())
                if len(correct)
                else float("nan")
            )
            incorrect_iou = (
                float(incorrect.groupby("patient_id")["top15_iou"].mean().mean())
                if len(incorrect)
                else float("nan")
            )

            test_summary_rows.append(
                {
                    "model": MODEL_DISPLAY_NAME,
                    "method": method_name,
                    "target_layers": "|".join(config["target_layers"]),
                    "target_resolutions": "|".join(
                        str(value) for value in config["target_resolutions"]
                    ),
                    "n_target_layers": int(config["n_target_layers"]),
                    "n_images": int(len(result)),
                    "n_patients": int(result["patient_id"].nunique()),
                    **overall,
                    **ci,
                    "correct_prediction_patient_balanced_top15_iou": correct_iou,
                    "incorrect_prediction_patient_balanced_top15_iou": incorrect_iou,
                }
            )

    test_cases = pd.concat(test_case_tables, ignore_index=True)
    test_summary = pd.DataFrame(test_summary_rows)

    if XAI_MAX_CASES == 0:
        expected_case_rows = EXPECTED_TEST_IMAGES * len(METHODS)
        if len(test_cases) != expected_case_rows:
            raise RuntimeError(
                "Combined XAI test row count mismatch. "
                f"Expected {expected_case_rows}, got {len(test_cases)}."
            )

    # Cross-method audit: classification output must be identical because CAM
    # method must not alter the classifier forward result.
    probability_spread = (
        test_cases.groupby("filename")["probability_malignant"].agg(
            lambda s: float(s.max() - s.min())
        )
    )
    if float(probability_spread.max()) > 1e-7:
        raise RuntimeError(
            "Classifier probabilities differ across XAI methods; "
            "the XAI methods must not alter classifier output."
        )

    cases_path = XAI_OUT / "test_xai_case_metrics.csv"
    summary_path = XAI_OUT / "test_xai_summary.csv"
    test_cases.to_csv(cases_path, index=False)
    test_summary.to_csv(summary_path, index=False)

    final_manifest = {
        "status": f"THYROIDXL_{MODEL_XAI_TAG}_XAI_ANATOMICAL_COORDINATE_FINAL_TEST",
        "protocol_version": "method_specific_anatomical_coordinates_v4",
        "model": MODEL_DISPLAY_NAME,
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": checkpoint_sha,
        "training_manifest": str(manifest_path),
        "freeze_file": str(freeze_path),
        "development_checkpoint_used": False,
        "configuration_source": "matched_architecture_fixed_16x16_configuration",
        "selected_methods": chosen,
        "same_target_layer_for_all_methods": False,
        "official_test_images": int(EXPECTED_TEST_IMAGES),
        "official_test_patients": int(EXPECTED_TEST_PATIENTS),
        "official_test_used_for_xai_layer_selection": False,
        "coordinate_space_for_localisation_metrics": "original_ultrasound",
        "cam_generation_space": "exact_padded_model_input",
        "padding_removed_after_cam_generation": True,
        "cam_retuned_after_test": False,
        "expert_mask_used_to_modify_cam": False,
        "yolo_mask_used_to_modify_cam": False,
        "bootstrap_unit": "patient",
        "bootstrap_samples": int(XAI_BOOTSTRAP_SAMPLES),
        "methods": test_summary_rows,
    }

    manifest_out = XAI_OUT / "xai_anatomical_final_manifest.json"
    manifest_out.write_text(
        json.dumps(final_manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print(f"{MODEL_DISPLAY_NAME.upper()} XAI FINAL SUMMARY")
    print("=" * 80)
    print(test_summary.to_string(index=False))
    print()
    print("Saved:", XAI_OUT)
    print("Case metrics:", cases_path)
    print("Summary:", summary_path)
    print("Manifest:", manifest_out)
    print("Final neural-network checkpoints used: 1")
    print("No CAM method/layer was selected or altered using held-out XAI performance.")


if __name__ == "__main__":
    xai_main()
