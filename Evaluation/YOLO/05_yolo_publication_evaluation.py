from __future__ import annotations

"""
ThyroidXL YOLOv8s-seg — SINGLE-FINAL-MODEL publication evaluation.

Purpose
-------
Evaluate exactly ONE neural-network checkpoint: the final YOLOv8s-seg model
refit on all 9,541 official-training images / 3,354 patients.

The evaluator:
- identifies the exact final YOLO checkpoint by the SHA256 recorded by the reviewed training notebook;
- optionally cross-checks a matching final training manifest when that sidecar file is present;
- never loads or evaluates the Fold-1 development checkpoint;
- verifies the frozen final-refit provenance before official-test access;
- uses the reviewed development-frozen inference protocol embedded below when the sidecar manifest is absent;
- uses a fixed, test-independent signed-score threshold of 0.0 as the primary
  hard-classification operating point;
- optionally reports a previously saved development-selected threshold from
  results/YOLO/yolo_evaluation_freeze.json if that JSON already exists and
  certifies that the official test was not used for selection;
- opens the official held-out 2,094-image / 739-patient cohort only after all
  model/protocol checks are complete;
- reports detection/segmentation, image-level classification, patient-level
  classification, bootstrap confidence intervals, and publication figures.

YOLO malignant ranking score
----------------------------
    max(malignant detection confidence) - max(benign detection confidence)

This is a signed detector-evidence score, NOT a calibrated probability.
Therefore 0.0 is the natural neutral fixed threshold. Ties are assigned benign:
    score > 0   -> malignant evidence > benign evidence
    score <= 0  -> benign evidence >= malignant evidence

No threshold, model, epoch, TTA setting, or protocol choice is optimized on the
official held-out cohort.
"""

import hashlib
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from huggingface_hub import get_token, hf_hub_download
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm
import ultralytics
from ultralytics import YOLO


# =============================================================================
# Frozen study constants
# =============================================================================

SEED = 42

EXPECTED_TRAIN_IMAGES = 9541
EXPECTED_TRAIN_PATIENTS = 3354
EXPECTED_TRAIN_BENIGN_PATIENTS = 2477
EXPECTED_TRAIN_MALIGNANT_PATIENTS = 877

EXPECTED_FOLD1_TRAIN_IMAGES = 7684
EXPECTED_FOLD1_TRAIN_PATIENTS = 2683
EXPECTED_FOLD1_VAL_IMAGES = 1857
EXPECTED_FOLD1_VAL_PATIENTS = 671

EXPECTED_TEST_IMAGES = 2094
EXPECTED_TEST_PATIENTS = 739
EXPECTED_TEST_BENIGN_PATIENTS = 386
EXPECTED_TEST_MALIGNANT_PATIENTS = 353

FINAL_STATUS = "THYROIDXL_YOLO_FINAL_OFFICIAL_TRAIN_REFIT_ONEFOLD_SELECTED"
DEFAULT_REPO_ID = "hunglc007/ThyroidXL"
DEFAULT_REPO_REVISION = "b15fe293bd74f1a8a4f05bf88bcdf06a1934125f"

# IMPORTANT: final-model identity is manifest-driven.
# Do not hard-code a checkpoint SHA or development-selected epoch here. The new
# 50-epoch development run may select a different epoch; after the final 9,541-image
# refit, its FINAL training manifest must record the exact checkpoint SHA and frozen
# protocol. Evaluation stops safely if that artifact is absent or ambiguous.

# Signed detector evidence has a natural neutral operating point at zero.
PRIMARY_IMAGE_THRESHOLD = 0.0
PRIMARY_PATIENT_THRESHOLD = 0.0

BOOTSTRAP_SAMPLES = 2000
DOWNLOAD_WORKERS = 4
VAL_BATCH = 8
VAL_WORKERS = 4
DEVICE = "0" if torch.cuda.is_available() else "cpu"

SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# Project / artifact discovery
# =============================================================================

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
        "Could not locate ThyroidXL project root. Expected a Models/ directory "
        "above this script, or set THYROIDXL_PROJECT_ROOT."
    )


ROOT = project_root()
MODELS_ROOT = ROOT / "Models" / "YOLOv8sSeg"
RESULTS_ROOT = ROOT / "results"
OUT = RESULTS_ROOT / "YOLO" / "FinalOfficialTest"
OUT.mkdir(parents=True, exist_ok=True)

OPTIONAL_DEV_FREEZE = RESULTS_ROOT / "YOLO" / "yolo_evaluation_freeze.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recursive_values(value, key):
    found = []
    if isinstance(value, dict):
        for current_key, child in value.items():
            if str(current_key) == str(key):
                found.append(child)
            found.extend(_recursive_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_recursive_values(child, key))
    return found


def _first_int_for_keys(value, keys):
    for key in keys:
        for candidate in _recursive_values(value, key):
            try:
                return int(candidate)
            except Exception:
                continue
    return None


def _first_string_for_keys(value, keys):
    for key in keys:
        for candidate in _recursive_values(value, key):
            if candidate is not None and str(candidate).strip():
                return str(candidate).strip()
    return None


def _manifest_checkpoint_sha(manifest: dict) -> str:
    value = (
        manifest.get("final_checkpoint_sha256")
        or manifest.get("checkpoint_sha256")
        or manifest.get("canonical_sha256")
    )
    if value is None:
        raise RuntimeError("Final YOLO manifest has no checkpoint SHA256.")
    value = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RuntimeError("Final YOLO manifest checkpoint SHA256 is invalid.")
    return value


def _validate_final_manifest(manifest: dict) -> str:
    """Validate final-refit provenance without hard-coding the selected epoch."""
    if manifest.get("status") != FINAL_STATUS:
        raise RuntimeError("Unexpected final YOLO manifest status.")
    if manifest.get("dataset") != "ThyroidXL":
        raise RuntimeError("Final YOLO manifest dataset is not ThyroidXL.")
    if manifest.get("official_test_accessed") is not False:
        raise RuntimeError(
            "Final YOLO training manifest does not certify official_test_accessed=false."
        )
    if int(manifest.get("training_images", -1)) != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError("Final YOLO was not trained on all 9,541 images.")
    if int(manifest.get("training_patients", -1)) != EXPECTED_TRAIN_PATIENTS:
        raise RuntimeError("Final YOLO was not trained on all 3,354 patients.")

    for key in (
        "internal_validation_used_in_final_refit",
        "independent_validation_used_in_final_refit",
        "validation_used_for_final_checkpoint_selection",
        "automatic_final_training_set_diagnostic_used_for_selection",
    ):
        if manifest.get(key) is not False:
            raise RuntimeError(f"Final YOLO manifest provenance check failed: {key}")

    selection = manifest.get("selection_source", {})
    expected_selection = {
        "validation_images": EXPECTED_FOLD1_VAL_IMAGES,
        "validation_patients": EXPECTED_FOLD1_VAL_PATIENTS,
        "training_images": EXPECTED_FOLD1_TRAIN_IMAGES,
        "training_patients": EXPECTED_FOLD1_TRAIN_PATIENTS,
        "patient_overlap": 0,
    }
    for key, expected in expected_selection.items():
        if int(selection.get(key, -1)) != expected:
            raise RuntimeError(
                f"Final YOLO selection-source {key} mismatch: "
                f"expected={expected}, observed={selection.get(key)!r}"
            )
    if selection.get("test_used_for_selection") is not False:
        raise RuntimeError("Final YOLO selection source does not certify untouched test.")

    selected_epoch = int(selection.get("selected_training_epoch_one_based", -1))
    if selected_epoch < 1:
        raise RuntimeError(
            "Final YOLO manifest has no valid development-selected training epoch."
        )

    training_cfg = manifest.get("training_configuration", {})
    if not isinstance(training_cfg, dict):
        raise RuntimeError("Final YOLO manifest training_configuration is invalid.")
    image_size = int(training_cfg.get("image_size", -1))
    if image_size <= 0:
        raise RuntimeError("Final YOLO manifest has no valid image_size.")

    inference_cfg = manifest.get("inference_protocol", {})
    if not isinstance(inference_cfg, dict):
        raise RuntimeError("Final YOLO manifest inference_protocol is invalid.")
    for key in ("confidence", "nms_iou", "max_detections"):
        if key not in inference_cfg:
            raise RuntimeError(f"Final YOLO manifest inference_protocol missing {key!r}.")

    return _manifest_checkpoint_sha(manifest)


def find_final_checkpoint_and_manifest():
    """
    Identify exactly one FINAL 9,541-image refit from its training manifest.

    This deliberately has NO embedded old-SHA fallback. That prevents a stale
    25-epoch final model from being silently evaluated after a new development run.
    """
    if not MODELS_ROOT.is_dir():
        raise FileNotFoundError(f"Missing YOLO model directory: {MODELS_ROOT}")

    manifest_matches = []
    validation_errors = []
    for path in ROOT.rglob("*.json"):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("status") != FINAL_STATUS:
            continue
        try:
            sha = _validate_final_manifest(payload)
        except Exception as exc:
            validation_errors.append(f"{path}: {exc}")
            continue
        manifest_matches.append((path.resolve(), payload, sha))

    if not manifest_matches:
        detail = "\n".join(validation_errors[:20])
        suffix = f"\nInvalid candidates:\n{detail}" if detail else ""
        raise RuntimeError(
            "No valid FINAL YOLO refit manifest was found. First complete the "
            "final refit on all 9,541 official-training images using the epoch "
            "selected by the frozen development run, and save its final manifest."
            + suffix
        )

    distinct_shas = sorted({sha for _, _, sha in manifest_matches})
    if len(distinct_shas) != 1:
        details = "\n".join(
            f"  {path} -> {sha}" for path, _, sha in manifest_matches
        )
        raise RuntimeError(
            "Multiple distinct FINAL YOLO model SHAs are present. Archive stale "
            f"final manifests before publication evaluation:\n{details}"
        )

    checkpoint_sha = distinct_shas[0]
    manifest_matches.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
    manifest_path, manifest, _ = manifest_matches[0]

    checkpoints = sorted(p.resolve() for p in MODELS_ROOT.rglob("*.pt") if p.is_file())
    if not checkpoints:
        raise RuntimeError(f"No YOLO .pt checkpoints found under {MODELS_ROOT}")
    checkpoint_matches = [
        path for path in checkpoints if sha256_file(path).lower() == checkpoint_sha
    ]
    if len(checkpoint_matches) != 1:
        observed = "\n".join(
            f"  {path}: {sha256_file(path).lower()}" for path in checkpoints
        )
        raise RuntimeError(
            "Could not locate exactly one final YOLO checkpoint matching the "
            f"manifest SHA {checkpoint_sha}. Observed:\n{observed}"
        )

    checkpoint_path = checkpoint_matches[0]
    repo_id = str(manifest.get("dataset_repo", DEFAULT_REPO_ID))
    revision = str(manifest.get("dataset_revision", DEFAULT_REPO_REVISION)).strip()
    if not revision:
        raise RuntimeError("Final YOLO provenance has no dataset_revision.")

    return (
        checkpoint_path,
        checkpoint_sha,
        manifest_path,
        manifest,
        "external_training_manifest_required",
        repo_id,
        revision,
    )


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


class HuggingFaceLazyCache:
    def __init__(
        self,
        repo_id: str,
        revision: str,
        token: str,
        max_attempts: int = 20,
    ):
        self.repo_id = str(repo_id)
        self.revision = str(revision)
        self.token = str(token)
        self.max_attempts = int(max_attempts)
        self._resolved_paths = {}

    def _download(self, repo_path: str, local_files_only: bool) -> Path:
        return Path(
            hf_hub_download(
                repo_id=self.repo_id,
                filename=repo_path,
                repo_type="dataset",
                revision=self.revision,
                token=self.token,
                local_files_only=local_files_only,
            )
        ).resolve()

    def fetch(self, repo_path: str) -> Path:
        repo_path = str(repo_path).replace("\\", "/").lstrip("/")

        remembered = self._resolved_paths.get(repo_path)
        if remembered is not None and Path(remembered).is_file():
            return Path(remembered)

        try:
            path = self._download(repo_path, local_files_only=True)
            self._resolved_paths[repo_path] = path
            return path
        except Exception:
            pass

        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                path = self._download(repo_path, local_files_only=False)
                self._resolved_paths[repo_path] = path
                return path
            except Exception as exc:
                last_error = exc
                text = repr(exc).lower()

                if "404" in text:
                    raise FileNotFoundError(repo_path) from exc
                if "401" in text or "403" in text:
                    raise PermissionError(
                        f"Hugging Face denied access while fetching {repo_path}."
                    ) from exc
                if attempt == self.max_attempts:
                    break

                wait = min(180.0, 5.0 * attempt)
                print(
                    f"Hugging Face access error for {repo_path}: {exc}\n"
                    f"Retrying in {wait:.0f}s ({attempt + 1}/{self.max_attempts})..."
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Could not fetch {repo_path} after {self.max_attempts} attempts."
        ) from last_error


# =============================================================================
# Dataset metadata / safety checks
# =============================================================================

def patient_id_from_filename(filename: str) -> str:
    stem = Path(str(filename)).stem
    match = re.match(r"^(\d+)(?:_|$)", stem)
    if match is None:
        raise ValueError(f"Cannot derive patient ID from filename: {filename}")
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
            raise RuntimeError(f"{path.name} has no {key!r} key.")

    mapping = {
        item["id"]: str(item.get("name", "")).strip()
        for item in coco.get("categories", [])
        if isinstance(item, dict) and "id" in item
    }

    image_id_to_filename = {}
    rows = []
    for item in coco["images"]:
        image_id = item["id"]
        filename = Path(str(item["file_name"])).name
        if image_id in image_id_to_filename:
            raise RuntimeError(f"Duplicate image ID: {image_id}")
        image_id_to_filename[image_id] = filename
        rows.append(
            {
                "image_id": image_id,
                "filename": filename,
                "patient_id": patient_id_from_filename(filename),
            }
        )

    by_image = {}
    for ann in coco["annotations"]:
        if not isinstance(ann, dict):
            continue
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")
        if image_id in image_id_to_filename and category_id is not None:
            by_image.setdefault(image_id, set()).add(category_id)

    labels = {}
    bad = []
    for image_id, filename in image_id_to_filename.items():
        values = {
            category_to_binary(cid, mapping)
            for cid in by_image.get(image_id, set())
        }
        values.discard(None)
        if len(values) != 1:
            bad.append((filename, sorted(map(str, by_image.get(image_id, set())))))
        else:
            labels[image_id] = next(iter(values))

    if len(labels) != len(rows):
        raise RuntimeError(
            "Could not derive one benign/malignant label for every image. "
            f"Resolved={len(labels)}/{len(rows)}; examples={bad[:5]}"
        )

    frame = pd.DataFrame(rows)
    frame["label"] = frame["image_id"].map(labels).astype(int)

    if frame["filename"].duplicated().any():
        raise RuntimeError(f"Duplicate filenames in {path.name}.")
    if int(frame.groupby("patient_id")["label"].nunique().max()) != 1:
        raise RuntimeError(f"A patient in {path.name} has inconsistent labels.")

    return frame.sort_values(["patient_id", "filename"]).reset_index(drop=True)


def load_official_metadata(remote_cache: HuggingFaceLazyCache):
    train = build_coco_frame(remote_cache.fetch("train/train_annotations.json"))

    if len(train) != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_TRAIN_IMAGES} train images, got {len(train)}")
    if train["patient_id"].nunique() != EXPECTED_TRAIN_PATIENTS:
        raise RuntimeError("Official training patient count mismatch.")

    train_patients = train[["patient_id", "label"]].drop_duplicates()
    train_counts = train_patients["label"].value_counts().sort_index().to_dict()
    if train_counts != {
        0: EXPECTED_TRAIN_BENIGN_PATIENTS,
        1: EXPECTED_TRAIN_MALIGNANT_PATIENTS,
    }:
        raise RuntimeError(f"Unexpected training patient class counts: {train_counts}")

    # First official-test access occurs here, after checkpoint/manifest/protocol freeze.
    test = build_coco_frame(remote_cache.fetch("test/test_annotations.json"))

    if len(test) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_TEST_IMAGES} test images, got {len(test)}")
    if test["patient_id"].nunique() != EXPECTED_TEST_PATIENTS:
        raise RuntimeError("Official test patient count mismatch.")

    test_patients = test[["patient_id", "label"]].drop_duplicates()
    test_counts = test_patients["label"].value_counts().sort_index().to_dict()
    if test_counts != {
        0: EXPECTED_TEST_BENIGN_PATIENTS,
        1: EXPECTED_TEST_MALIGNANT_PATIENTS,
    }:
        raise RuntimeError(f"Unexpected test patient class counts: {test_counts}")

    patient_overlap = set(train["patient_id"]) & set(test["patient_id"])
    filename_overlap = set(train["filename"]) & set(test["filename"])
    if patient_overlap:
        raise RuntimeError(f"Official train/test patient overlap: {sorted(patient_overlap)[:10]}")
    if filename_overlap:
        raise RuntimeError(f"Official train/test filename overlap: {sorted(filename_overlap)[:10]}")

    return train, test


def read_binary_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")

    mask = np.squeeze(np.asarray(mask))
    if mask.ndim == 3:
        if mask.shape[-1] in (3, 4):
            colour = mask[..., :3]
            if (
                np.array_equal(colour[..., 0], colour[..., 1])
                and np.array_equal(colour[..., 0], colour[..., 2])
            ):
                mask = colour[..., 0]
            else:
                mask = np.max(colour, axis=-1)
        else:
            raise RuntimeError(f"Unexpected 3-D mask shape: {mask.shape}")

    if mask.ndim != 2:
        raise RuntimeError(f"Expected 2-D mask, got {mask.shape}")

    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        raise RuntimeError(f"Empty expert nodule mask: {mask_path.name}")
    return binary


def cache_test_pairs(
    test: pd.DataFrame,
    remote_cache: HuggingFaceLazyCache,
    workers: int = DOWNLOAD_WORKERS,
):
    names = test["filename"].tolist()

    def fetch_pair(filename):
        remote_cache.fetch(f"test/images/{filename}")
        remote_cache.fetch(f"test/masks/{filename}")
        return filename

    failures = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(fetch_pair, name): name for name in names}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Caching official test image/mask pairs",
            unit="pair",
        ):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((name, repr(exc)))

    if failures:
        raise RuntimeError(f"{len(failures)} cache failures. Examples: {failures[:10]}")


# =============================================================================
# YOLO segmentation-label conversion for standard Ultralytics metrics
# =============================================================================

def largest_external_contour(mask_binary):
    contours, _ = cv2.findContours(
        mask_binary.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = [
        c for c in contours
        if c is not None and len(c) >= 3 and cv2.contourArea(c) > 0
    ]
    if not contours:
        raise RuntimeError("No valid contour found.")
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[0], contours


def polygon_reconstruction_iou(mask_binary, polygon_xy):
    reconstruction = np.zeros_like(mask_binary, dtype=np.uint8)
    cv2.fillPoly(
        reconstruction,
        [np.round(polygon_xy).astype(np.int32).reshape(-1, 1, 2)],
        1,
    )
    gt = mask_binary.astype(bool)
    pred = reconstruction.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return float(inter / union) if union else 0.0


def simplify_contour_with_fidelity(mask_binary, contour):
    perimeter = cv2.arcLength(contour, closed=True)
    best = None
    for fraction in (0.0020, 0.0010, 0.0005, 0.00025, 0.0):
        candidate = (
            contour
            if fraction == 0.0
            else cv2.approxPolyDP(
                contour,
                epsilon=fraction * perimeter,
                closed=True,
            )
        )
        points = candidate.reshape(-1, 2)
        if len(points) < 3:
            continue
        iou = polygon_reconstruction_iou(mask_binary, points)
        best = (points, iou, fraction)
        if iou >= 0.98:
            break
    if best is None:
        raise RuntimeError("Could not construct a valid segmentation polygon.")
    return best


def hardlink_or_copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    try:
        os.link(source, destination)
        return
    except OSError:
        pass

    try:
        os.symlink(source, destination)
        return
    except OSError:
        pass

    shutil.copy2(source, destination)


def build_yolo_eval_dataset(
    test: pd.DataFrame,
    remote_cache: HuggingFaceLazyCache,
    root: Path,
):
    images_dir = root / "images" / "val"
    labels_dir = root / "labels" / "val"

    if root.exists():
        shutil.rmtree(root)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    audit_rows = []

    for row in tqdm(
        test.itertuples(index=False),
        total=len(test),
        desc="Building YOLO official-test labels",
        unit="img",
    ):
        filename = str(row.filename)
        class_id = int(row.label)

        source_image = remote_cache.fetch(f"test/images/{filename}")
        source_mask = remote_cache.fetch(f"test/masks/{filename}")

        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unreadable image: {filename}")
        mask_binary = read_binary_mask(source_mask)

        if image.shape[:2] != mask_binary.shape:
            raise RuntimeError(
                f"Image/mask shape mismatch for {filename}: "
                f"{image.shape[:2]} vs {mask_binary.shape}"
            )

        h, w = mask_binary.shape
        main_contour, all_contours = largest_external_contour(mask_binary)

        main_mask = np.zeros_like(mask_binary)
        cv2.fillPoly(main_mask, [main_contour], 1)
        retained_fraction = float(
            np.logical_and(mask_binary > 0, main_mask > 0).sum()
            / max(1, (mask_binary > 0).sum())
        )

        points, reconstruction_iou, epsilon_fraction = simplify_contour_with_fidelity(
            mask_binary, main_contour
        )

        normalized = points.astype(np.float64)
        normalized[:, 0] = np.clip(normalized[:, 0] / float(w), 0.0, 1.0)
        normalized[:, 1] = np.clip(normalized[:, 1] / float(h), 0.0, 1.0)

        yolo_line = f"{class_id} " + " ".join(
            f"{v:.8f}" for v in normalized.reshape(-1)
        )

        target_image = images_dir / filename
        target_label = labels_dir / f"{Path(filename).stem}.txt"
        hardlink_or_copy(source_image, target_image)
        target_label.write_text(yolo_line + "\n", encoding="utf-8")

        audit_rows.append(
            {
                "filename": filename,
                "patient_id": str(row.patient_id),
                "label": class_id,
                "expert_mask_fraction": float(mask_binary.mean()),
                "external_components": len(all_contours),
                "largest_component_retained_fraction": retained_fraction,
                "polygon_points": len(normalized),
                "polygon_reconstruction_iou": reconstruction_iou,
                "epsilon_fraction": epsilon_fraction,
            }
        )

    audit = pd.DataFrame(audit_rows)
    if (audit["largest_component_retained_fraction"] < 0.95).any():
        raise RuntimeError("At least one test mask retains <95% foreground.")
    if (audit["polygon_reconstruction_iou"] < 0.95).any():
        raise RuntimeError("At least one test polygon reconstructs with IoU <0.95.")

    image_list = root / "val_images.txt"
    image_list.write_text(
        "\n".join(str(p.absolute()) for p in sorted(images_dir.iterdir())),
        encoding="utf-8",
    )

    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": str(image_list),
                "val": str(image_list),
                "nc": 2,
                "names": ["benign", "malignant"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return yaml_path, audit


# =============================================================================
# Metrics
# =============================================================================

def safe_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_auprc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def specificity_score(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    return float(tn / (tn + fp)) if (tn + fp) else float("nan")


def binary_metrics(labels, scores, threshold, *, strict_greater=False):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if strict_greater:
        predictions = (scores > float(threshold)).astype(np.int64)
    else:
        predictions = (scores >= float(threshold)).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "threshold_comparison": ">" if strict_greater else ">=",
        "n": int(len(labels)),
        "roc_auc": safe_auc(labels, scores),
        "auprc": safe_auprc(labels, scores),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity_malignant": float(
            recall_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "specificity_benign": specificity_score(labels, predictions),
        "precision_malignant": float(
            precision_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "f1_malignant": float(
            f1_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_patient_metrics(
    patient_df: pd.DataFrame,
    threshold: float,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = SEED,
    *,
    strict_greater: bool = False,
):
    labels_all = patient_df["label"].to_numpy(dtype=int)
    scores_all = patient_df["malignant_score"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    keys = [
        "roc_auc",
        "auprc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity_malignant",
        "specificity_benign",
        "f1_malignant",
        "mcc",
    ]
    values = {key: [] for key in keys}

    for _ in range(int(samples)):
        idx = rng.integers(0, len(patient_df), size=len(patient_df))
        labels = labels_all[idx]
        scores = scores_all[idx]
        if np.unique(labels).size < 2:
            continue
        result = binary_metrics(
            labels,
            scores,
            threshold,
            strict_greater=strict_greater,
        )
        for key in keys:
            value = result[key]
            if np.isfinite(value):
                values[key].append(float(value))

    output = {}
    for key, arr in values.items():
        if arr:
            a = np.asarray(arr, dtype=float)
            output[key] = [
                float(np.quantile(a, 0.025)),
                float(np.quantile(a, 0.975)),
            ]
    return output


def cluster_bootstrap_mean(
    image_df: pd.DataFrame,
    column: str,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = SEED,
):
    patient_ids = image_df["patient_id"].drop_duplicates().tolist()
    grouped = {
        pid: group[column].to_numpy(dtype=float)
        for pid, group in image_df.groupby("patient_id")
    }
    rng = np.random.default_rng(seed)
    values = []

    for _ in range(int(samples)):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        pieces = [grouped[pid] for pid in sampled]
        values.append(float(np.concatenate(pieces).mean()))

    arr = np.asarray(values, dtype=float)
    return [
        float(np.quantile(arr, 0.025)),
        float(np.quantile(arr, 0.975)),
    ]


# =============================================================================
# Prediction / segmentation
# =============================================================================

def class_ids(model):
    names = model.names
    if not isinstance(names, dict):
        names = {i: str(v) for i, v in enumerate(names)}
    names = {int(k): str(v).strip().lower() for k, v in names.items()}

    benign = [k for k, v in names.items() if "benign" in v]
    malignant = [k for k, v in names.items() if "malignant" in v]
    if len(benign) != 1 or len(malignant) != 1:
        raise RuntimeError(f"Expected one benign and one malignant class; names={names}")
    return int(benign[0]), int(malignant[0])


def mask_metrics(pred_mask, gt_mask):
    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = float(inter / union) if union else 0.0
    denom = pred.sum() + gt.sum()
    dice = float(2 * inter / denom) if denom else 0.0
    return iou, dice


def evaluate_images(
    model,
    test,
    remote_cache,
    image_size: int,
    retrieval_confidence: float,
    nms_iou: float,
    max_detections: int,
    augment: bool,
):
    benign_id, malignant_id = class_ids(model)
    rows = []

    for row in tqdm(
        test.itertuples(index=False),
        total=len(test),
        desc="YOLO final official-test inference",
        unit="img",
    ):
        filename = str(row.filename)
        image_path = remote_cache.fetch(f"test/images/{filename}")
        mask_path = remote_cache.fetch(f"test/masks/{filename}")
        gt_mask = read_binary_mask(mask_path).astype(bool)

        result = model.predict(
            source=str(image_path),
            imgsz=int(image_size),
            conf=float(retrieval_confidence),
            iou=float(nms_iou),
            max_det=int(max_detections),
            device=DEVICE,
            augment=bool(augment),
            retina_masks=True,
            verbose=False,
        )[0]

        boxes = result.boxes
        p_b = 0.0
        p_m = 0.0
        has_detection = bool(boxes is not None and len(boxes) > 0)
        top_class = None
        top_conf = 0.0
        top_mask_iou = 0.0
        top_mask_dice = 0.0
        pred_mask_fraction = 0.0

        if has_detection:
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            confs = boxes.conf.detach().cpu().numpy().astype(float)

            benign_mask = classes == benign_id
            malignant_mask = classes == malignant_id
            if benign_mask.any():
                p_b = float(confs[benign_mask].max())
            if malignant_mask.any():
                p_m = float(confs[malignant_mask].max())

            top_idx = int(np.argmax(confs))
            top_class = int(classes[top_idx])
            top_conf = float(confs[top_idx])

            if result.masks is not None and len(result.masks.data) > top_idx:
                pred = (
                    result.masks.data[top_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                if pred.shape != gt_mask.shape:
                    # With retina_masks=True Ultralytics should return masks in
                    # original-image HxW. Never blindly resize a letterboxed mask:
                    # rasterize the original-coordinate polygon as a safe fallback.
                    segments = result.masks.xy
                    if len(segments) <= top_idx or len(segments[top_idx]) < 3:
                        raise RuntimeError(
                            f"YOLO native-mask geometry mismatch for {filename}: "
                            f"prediction={pred.shape}, expert={gt_mask.shape}, and "
                            "no usable original-coordinate polygon is available."
                        )
                    raster = np.zeros(gt_mask.shape, dtype=np.uint8)
                    polygon = np.rint(np.asarray(segments[top_idx], dtype=np.float32)).astype(np.int32)
                    polygon[:, 0] = np.clip(polygon[:, 0], 0, gt_mask.shape[1] - 1)
                    polygon[:, 1] = np.clip(polygon[:, 1], 0, gt_mask.shape[0] - 1)
                    cv2.fillPoly(raster, [polygon.reshape(-1, 1, 2)], 1)
                    pred = raster.astype(np.float32)
                pred = pred > 0.5
                top_mask_iou, top_mask_dice = mask_metrics(pred, gt_mask)
                pred_mask_fraction = float(pred.mean())

        signed_score = float(p_m - p_b)

        rows.append(
            {
                "filename": filename,
                "patient_id": str(row.patient_id),
                "label": int(row.label),
                "has_detection": int(has_detection),
                "n_detections": int(len(boxes)) if boxes is not None else 0,
                "benign_detection_score": p_b,
                "malignant_detection_score": p_m,
                "malignant_score": signed_score,
                "prediction_at_zero": int(signed_score > 0.0),
                "top_detection_class": top_class,
                "top_detection_confidence": top_conf,
                "expert_mask_fraction": float(gt_mask.mean()),
                "predicted_mask_fraction": pred_mask_fraction,
                "mask_iou": top_mask_iou,
                "mask_dice": top_mask_dice,
                "segmentation_geometry": "original_ultrasound",
            }
        )

    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_TEST_IMAGES:
        raise RuntimeError("Final YOLO inference row count mismatch.")
    if result["patient_id"].nunique() != EXPECTED_TEST_PATIENTS:
        raise RuntimeError("Final YOLO inference patient count mismatch.")
    return result


def aggregate_patients(image_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient_id, group in image_df.groupby("patient_id", sort=False):
        labels = group["label"].unique()
        if len(labels) != 1:
            raise RuntimeError(f"Patient {patient_id} has inconsistent labels: {labels}")

        rows.append(
            {
                "patient_id": str(patient_id),
                "label": int(labels[0]),
                "n_frames": int(len(group)),
                "malignant_score": float(group["malignant_score"].mean()),
                "detection_coverage": float(group["has_detection"].mean()),
                "mean_expert_mask_fraction": float(group["expert_mask_fraction"].mean()),
                "mean_mask_iou": float(group["mask_iou"].mean()),
                "mean_mask_dice": float(group["mask_dice"].mean()),
            }
        )
    return pd.DataFrame(rows)


def segmentation_summary(image_df: pd.DataFrame):
    all_cases = image_df.copy()
    detected = image_df[image_df["has_detection"].astype(int) == 1].copy()

    def one(frame):
        if len(frame) == 0:
            return None
        return {
            "n_images": int(len(frame)),
            "mean_mask_iou": float(frame["mask_iou"].fillna(0.0).mean()),
            "median_mask_iou": float(frame["mask_iou"].fillna(0.0).median()),
            "mean_mask_dice": float(frame["mask_dice"].fillna(0.0).mean()),
            "median_mask_dice": float(frame["mask_dice"].fillna(0.0).median()),
        }

    return {
        "geometry": "original_ultrasound",
        "coordinate_policy": (
            "Custom inference uses retina_masks=True so Ultralytics returns the "
            "selected mask in native original-ultrasound HxW. A polygon-raster "
            "fallback is used only if a shape mismatch is detected; blind resizing "
            "of a letterboxed mask is prohibited."
        ),
        "detection_coverage": float(image_df["has_detection"].mean()),
        "end_to_end_all_images": one(all_cases),
        "end_to_end_benign": one(all_cases[all_cases["label"] == 0]),
        "end_to_end_malignant": one(all_cases[all_cases["label"] == 1]),
        "detected_only_all": one(detected),
        "detected_only_benign": one(detected[detected["label"] == 0]),
        "detected_only_malignant": one(detected[detected["label"] == 1]),
        "note": (
            "End-to-end Dice/IoU assigns 0 to images with no usable YOLO mask. "
            "Detected-only values are reported separately."
        ),
    }


def ultralytics_metrics_record(metrics):
    record = {
        "box_precision": float(metrics.box.mp),
        "box_recall": float(metrics.box.mr),
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
        "mask_precision": float(metrics.seg.mp),
        "mask_recall": float(metrics.seg.mr),
        "mask_map50": float(metrics.seg.map50),
        "mask_map50_95": float(metrics.seg.map),
    }

    box_maps = getattr(metrics.box, "maps", None)
    seg_maps = getattr(metrics.seg, "maps", None)
    if box_maps is not None and len(box_maps) >= 2:
        record["benign_box_map50_95"] = float(box_maps[0])
        record["malignant_box_map50_95"] = float(box_maps[1])
    if seg_maps is not None and len(seg_maps) >= 2:
        record["benign_mask_map50_95"] = float(seg_maps[0])
        record["malignant_mask_map50_95"] = float(seg_maps[1])
    return record


# =============================================================================
# Figures
# =============================================================================

def save_confusion_matrix(metrics: dict, path: Path, title: str):
    matrix = np.array(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        dtype=int,
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Benign", "Malignant"])
    ax.set_yticks([0, 1], labels=["Benign", "Malignant"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_figures(image_df: pd.DataFrame, patient_df: pd.DataFrame, primary_metrics: dict):
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(
        patient_df["label"],
        patient_df["malignant_score"],
        ax=ax,
        name="YOLOv8s-seg",
    )
    ax.set_title("ThyroidXL held-out patient ROC")
    fig.tight_layout()
    fig.savefig(OUT / "patient_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(
        patient_df["label"],
        patient_df["malignant_score"],
        ax=ax,
        name="YOLOv8s-seg",
    )
    ax.set_title("ThyroidXL held-out patient precision-recall")
    fig.tight_layout()
    fig.savefig(OUT / "patient_precision_recall_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    save_confusion_matrix(
        primary_metrics,
        OUT / "patient_confusion_matrix_threshold_0.png",
        "YOLOv8s-seg patient confusion matrix (score > 0)",
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(image_df["mask_dice"].to_numpy(dtype=float), bins=30)
    ax.set_xlabel("End-to-end mask Dice")
    ax.set_ylabel("Images")
    ax.set_title("YOLOv8s-seg held-out segmentation Dice")
    fig.tight_layout()
    fig.savefig(OUT / "segmentation_dice_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main: ONE MODEL ONLY
# =============================================================================

def main():
    print("=" * 80)
    print("THYROIDXL YOLOv8s-seg — SINGLE FINAL MODEL PUBLICATION EVALUATION")
    print("=" * 80)
    print("Project root:", ROOT)
    print()

    (
        checkpoint_path,
        checkpoint_sha,
        manifest_path,
        manifest,
        manifest_source,
        repo_id,
        revision,
    ) = find_final_checkpoint_and_manifest()

    training_cfg = manifest.get("training_configuration", {})
    inference_cfg = manifest.get("inference_protocol", {})

    image_size = int(training_cfg.get("image_size", 640))
    retrieval_confidence = float(inference_cfg.get("confidence", 0.001))
    nms_iou = float(inference_cfg.get("nms_iou", 0.70))
    max_detections = int(inference_cfg.get("max_detections", 300))
    augment = bool(inference_cfg.get("tta_recommended", False))

    # No secondary development-threshold artifact is loaded here. The publication
    # evaluator is intentionally primary-only for YOLO classification, using the
    # fixed neutral signed-score rule score > 0.0. This also prevents stale
    # threshold artifacts from earlier development runs being applied to the new
    # final refit.
    optional_dev = None

    # All model/protocol decisions are fixed before official test metadata access.
    token = resolve_hf_token()
    remote_cache = HuggingFaceLazyCache(
        repo_id=repo_id,
        revision=revision,
        token=token,
    )

    # Load EXACTLY ONE neural-network checkpoint.
    model = YOLO(str(checkpoint_path))
    benign_id, malignant_id = class_ids(model)

    print("Final checkpoint:", checkpoint_path)
    print("Checkpoint SHA256:", checkpoint_sha)
    print("Training manifest:", manifest_path if manifest_path is not None else "not present locally")
    print("Provenance source:", manifest_source)
    print("Models loaded: 1")
    print("Development checkpoint loaded: NO")
    print(f"YOLO classes: benign={benign_id}, malignant={malignant_id}")
    print("Image size:", image_size)
    print("Retrieval confidence:", retrieval_confidence)
    print("NMS IoU:", nms_iou)
    print("Max detections:", max_detections)
    print("Frozen TTA/augment:", augment)
    print("Primary signed-score threshold: score > 0.0 (ties benign)")
    print("Segmentation metrics: ORIGINAL ultrasound coordinates (retina_masks=True)")
    print("Ultralytics version:", ultralytics.__version__)
    if optional_dev is not None:
        print("Optional development-frozen thresholds found:", optional_dev)
    else:
        print("Optional development-frozen thresholds: not present")
    print("Official test opened so far: NO")
    print()

    official_train, test = load_official_metadata(remote_cache)

    print("Official held-out cohort:", len(test), "images /", test["patient_id"].nunique(), "patients")
    print("Train/test patient overlap: 0")
    print("Train/test filename overlap: 0")
    print("No test-set optimisation is performed.")
    print()

    cache_test_pairs(test, remote_cache)

    yolo_dataset_root = OUT / "_yolo_eval_dataset"
    yaml_path, polygon_audit = build_yolo_eval_dataset(
        test,
        remote_cache,
        yolo_dataset_root,
    )
    polygon_audit.to_csv(OUT / "polygon_conversion_audit.csv", index=False)

    standard_val = model.val(
        data=str(yaml_path),
        split="val",
        imgsz=image_size,
        batch=VAL_BATCH,
        rect=False,
        conf=retrieval_confidence,
        iou=nms_iou,
        max_det=max_detections,
        device=DEVICE,
        workers=VAL_WORKERS,
        augment=augment,
        plots=True,
        project=str(OUT / "ultralytics"),
        name="standard_metrics",
        exist_ok=True,
        verbose=True,
    )
    standard_metrics = ultralytics_metrics_record(standard_val)

    image_df = evaluate_images(
        model=model,
        test=test,
        remote_cache=remote_cache,
        image_size=image_size,
        retrieval_confidence=retrieval_confidence,
        nms_iou=nms_iou,
        max_detections=max_detections,
        augment=augment,
    )
    patient_df = aggregate_patients(image_df)

    # Primary fixed, test-independent neutral threshold.
    image_metrics_primary = binary_metrics(
        image_df["label"], image_df["malignant_score"], PRIMARY_IMAGE_THRESHOLD,
        strict_greater=True,
    )
    patient_metrics_primary = binary_metrics(
        patient_df["label"], patient_df["malignant_score"], PRIMARY_PATIENT_THRESHOLD,
        strict_greater=True,
    )

    image_df["prediction_primary"] = (
        image_df["malignant_score"] > PRIMARY_IMAGE_THRESHOLD
    ).astype(int)
    image_df["correct_primary"] = (
        image_df["prediction_primary"] == image_df["label"]
    ).astype(int)

    patient_df["prediction_primary"] = (
        patient_df["malignant_score"] > PRIMARY_PATIENT_THRESHOLD
    ).astype(int)
    patient_df["prediction_at_zero"] = patient_df["prediction_primary"].astype(int)
    patient_df["correct_primary"] = (
        patient_df["prediction_primary"] == patient_df["label"]
    ).astype(int)

    secondary = None
    if optional_dev is not None:
        image_threshold = optional_dev["image_threshold"]
        patient_threshold = optional_dev["patient_threshold"]

        image_metrics_secondary = binary_metrics(
            image_df["label"], image_df["malignant_score"], image_threshold
        )
        patient_metrics_secondary = binary_metrics(
            patient_df["label"], patient_df["malignant_score"], patient_threshold
        )

        image_df["prediction_development_threshold"] = (
            image_df["malignant_score"] >= image_threshold
        ).astype(int)
        patient_df["prediction_development_threshold"] = (
            patient_df["malignant_score"] >= patient_threshold
        ).astype(int)

        secondary = {
            "source": optional_dev,
            "image_level": image_metrics_secondary,
            "patient_level": patient_metrics_secondary,
            "patient_bootstrap_95_ci": bootstrap_patient_metrics(
                patient_df,
                threshold=patient_threshold,
                seed=SEED + 1000,
            ),
        }

    image_df.to_csv(OUT / "test_image_predictions.csv", index=False)
    patient_df.to_csv(OUT / "test_patient_predictions.csv", index=False)

    segmentation = segmentation_summary(image_df)
    primary_ci = bootstrap_patient_metrics(
        patient_df,
        threshold=PRIMARY_PATIENT_THRESHOLD,
        seed=SEED,
        strict_greater=True,
    )
    seg_dice_ci = cluster_bootstrap_mean(
        image_df,
        "mask_dice",
        seed=SEED + 2000,
    )
    seg_iou_ci = cluster_bootstrap_mean(
        image_df,
        "mask_iou",
        seed=SEED + 3000,
    )

    save_figures(image_df, patient_df, patient_metrics_primary)

    summary = {
        "status": "THYROIDXL_YOLO_SINGLE_FINAL_MODEL_OFFICIAL_TEST",
        "evaluation_model_count": 1,
        "development_checkpoint_loaded": False,
        "official_test_accessed": True,
        "official_test_used_for_tuning": False,
        "custom_mask_inference": {"retina_masks": True, "geometry": "original_ultrasound"},
        "dataset": "ThyroidXL",
        "dataset_repo": repo_id,
        "dataset_revision": revision,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "training_manifest": str(manifest_path) if manifest_path is not None else None,
        "provenance_source": manifest_source,
        "ultralytics_version": str(ultralytics.__version__),
        "official_training_images": EXPECTED_TRAIN_IMAGES,
        "official_training_patients": EXPECTED_TRAIN_PATIENTS,
        "official_test_images": EXPECTED_TEST_IMAGES,
        "official_test_patients": EXPECTED_TEST_PATIENTS,
        "train_test_patient_overlap": 0,
        "train_test_filename_overlap": 0,
        "score_definition": (
            "max malignant detection confidence - max benign detection confidence"
        ),
        "score_note": (
            "Signed detector evidence is an operating/ranking score, not a calibrated probability."
        ),
        "reporting_policy": {
            "primary_discrimination": ["ROC-AUC", "AUPRC"],
            "primary_image_threshold": PRIMARY_IMAGE_THRESHOLD,
            "primary_patient_threshold": PRIMARY_PATIENT_THRESHOLD,
            "primary_threshold_reason": (
                "fixed neutral signed-score threshold; score > 0 is malignant and ties/no-detections are benign; chosen independently of official test"
            ),
            "patient_aggregation": "mean image signed malignant score across patient frames",
            "optional_development_thresholds_present": optional_dev is not None,
            "zero_threshold_comparator": ">",
            "zero_score_tie_policy": "benign",
        },
        "inference": {
            "image_size": image_size,
            "retrieval_confidence": retrieval_confidence,
            "nms_iou": nms_iou,
            "max_detections": max_detections,
            "augment": augment,
        },
        "ultralytics_detection_segmentation": standard_metrics,
        "image_level_classification_primary": image_metrics_primary,
        "patient_level_classification_primary": patient_metrics_primary,
        "patient_level_bootstrap_95_ci_primary": primary_ci,
        "development_threshold_secondary": secondary,
        "case_level_segmentation": segmentation,
        "segmentation_bootstrap_95_ci_patient_cluster": {
            "mean_mask_dice": seg_dice_ci,
            "mean_mask_iou": seg_iou_ci,
        },
    }

    (OUT / "test_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("FINAL YOLO EVALUATION COMPLETE")
    print("=" * 80)
    print("Models loaded: 1")
    print("Development checkpoint loaded: NO")
    print("Output directory:", OUT)
    print()
    print("STANDARD YOLO DETECTION / SEGMENTATION")
    print(json.dumps(standard_metrics, indent=2))
    print()
    print("IMAGE-LEVEL CLASSIFICATION — FIXED NEUTRAL RULE score > 0.0")
    print(json.dumps(image_metrics_primary, indent=2))
    print()
    print("PATIENT-LEVEL CLASSIFICATION — FIXED NEUTRAL RULE score > 0.0")
    print(json.dumps(patient_metrics_primary, indent=2))
    if secondary is not None:
        print()
        print("DEVELOPMENT-FROZEN THRESHOLD — SECONDARY")
        print(json.dumps(secondary["patient_level"], indent=2))
    print()
    print("SEGMENTATION")
    print(json.dumps(segmentation, indent=2))


if __name__ == "__main__":
    main()
