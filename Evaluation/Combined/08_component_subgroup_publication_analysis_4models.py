"""
ThyroidXL — canonical four-model subgroup analysis.

Uses one common, leakage-safe nodule-size definition for all FINAL models:
MobileNetV3-Large, EfficientNet-B3, ConvNeXt-Tiny and YOLOv8s-seg.

Size boundaries are derived ONLY from official TRAIN expert-mask fractions and
then applied unchanged to the official held-out cohort. Classification and
segmentation subgroup uncertainty remains patient-aware.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import get_token, hf_hub_download
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm


# =============================================================================
# Frozen study constants
# =============================================================================

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO_ID = "hunglc007/ThyroidXL"
REPO_REVISION = "b15fe293bd74f1a8a4f05bf88bcdf06a1934125f"

EXPECTED_TRAIN_IMAGES = 9541
EXPECTED_TRAIN_PATIENTS = 3354
EXPECTED_TEST_IMAGES = 2094
EXPECTED_TEST_PATIENTS = 739

MASK_WORKERS = 4
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# Project / result discovery
# =============================================================================

def find_project_root() -> Path:
    env = os.environ.get("THYROIDXL_PROJECT_ROOT")
    if env:
        candidate = Path(env).expanduser().resolve()
        if candidate.is_dir():
            return candidate

    for candidate in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (
            (candidate / "Models").is_dir()
            and (candidate / "results").is_dir()
        ):
            return candidate

    raise RuntimeError(
        "Could not locate ThyroidXL project root. Expected Models/ and results/ "
        "above this script, or set THYROIDXL_PROJECT_ROOT."
    )


ROOT = find_project_root()
RESULTS = ROOT / "results"
OUT = RESULTS / "Combined" / "Subgroups" / "FinalOfficialTest"
OUT.mkdir(parents=True, exist_ok=True)


def unique_result(
    filename: str,
    model_token: str,
    preferred: Path | None = None,
) -> Path:
    if preferred is not None and preferred.is_file():
        return preferred.resolve()

    matches = [
        path.resolve()
        for path in RESULTS.rglob(filename)
        if (
            path.is_file()
            and model_token.lower() in str(path).lower()
            and "validation" not in str(path).lower()
            and "development" not in str(path).lower()
        )
    ]

    if len(matches) == 1:
        return matches[0]

    # Prefer paths that explicitly look final/test if that resolves ambiguity.
    strong = [
        path
        for path in matches
        if (
            "final" in str(path).lower()
            or f"{os.sep}test{os.sep}" in str(path).lower()
            or "officialtest" in str(path).lower()
        )
    ]

    if len(strong) == 1:
        return strong[0]

    raise RuntimeError(
        f"Expected exactly one final {model_token} {filename!r}; found "
        f"{len(matches)}:\n" + "\n".join(str(path) for path in matches)
    )


def recursive_contains_full_train_counts(value) -> bool:
    if isinstance(value, dict):
        image_keys = (
            "training_images",
            "official_training_images",
            "train_images",
        )
        patient_keys = (
            "training_patients",
            "official_training_patients",
            "train_patients",
        )

        image_value = next(
            (value[key] for key in image_keys if key in value),
            None,
        )
        patient_value = next(
            (value[key] for key in patient_keys if key in value),
            None,
        )

        if image_value is not None and patient_value is not None:
            try:
                if (
                    int(image_value) == EXPECTED_TRAIN_IMAGES
                    and int(patient_value) == EXPECTED_TRAIN_PATIENTS
                ):
                    return True
            except Exception:
                pass

        return any(
            recursive_contains_full_train_counts(child)
            for child in value.values()
        )

    if isinstance(value, list):
        return any(
            recursive_contains_full_train_counts(child)
            for child in value
        )

    return False


def verify_final_result_provenance(
    csv_path: Path,
    model_token: str,
):
    """
    Tie provenance to the EXACT prediction CSV being analysed.

    The sibling test_summary.json must certify:
    - full final training: 9,541 images / 3,354 patients;
    - held-out evaluation: 2,094 images / 739 patients;
    - no test-set tuning;
    - a concrete final checkpoint SHA256.

    This prevents an unrelated/stale JSON elsewhere in results/ from making
    an old Fold-1 prediction CSV look publication-final.
    """
    summary_path = csv_path.parent / "test_summary.json"

    if not summary_path.is_file():
        raise RuntimeError(
            f"PUBLICATION SAFETY STOP: {model_token} prediction CSV has no "
            f"sibling test_summary.json:\n  {csv_path}"
        )

    try:
        payload = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not read provenance summary: {summary_path}"
        ) from exc

    if not recursive_contains_full_train_counts(payload):
        raise RuntimeError(
            f"PUBLICATION SAFETY STOP: {model_token} sibling summary does not "
            "certify a model trained on all 9,541 images / 3,354 patients."
        )

    if int(payload.get("official_test_images", -1)) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"{model_token} test summary does not certify 2,094 test images."
        )
    if int(payload.get("official_test_patients", -1)) != EXPECTED_TEST_PATIENTS:
        raise RuntimeError(
            f"{model_token} test summary does not certify 739 test patients."
        )

    if payload.get("official_test_used_for_tuning") is not False:
        raise RuntimeError(
            f"{model_token} test summary does not explicitly certify "
            "official_test_used_for_tuning=false."
        )

    checkpoint_sha = str(
        payload.get("checkpoint_sha256", "")
    ).strip().lower()

    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha):
        raise RuntimeError(
            f"{model_token} test summary has no valid checkpoint SHA256."
        )

    status = str(payload.get("status", "")).upper()
    expected_token = str(model_token).replace("-", "").replace("_", "").upper()
    normalized_status = status.replace("-", "").replace("_", "")
    # Allow conventional aliases while still requiring model identity + FINAL.
    aliases = {
        "MOBILENETV3": "MOBILENET",
        "EFFICIENTNETB3": "EFFICIENTNETB3",
        "CONVNEXTTINY": "CONVNEXTTINY",
        "YOLOV8SSEG": "YOLO",
    }
    token = aliases.get(expected_token, expected_token)
    if token not in normalized_status or "FINAL" not in normalized_status:
        raise RuntimeError(
            f"{model_token} sibling summary is not marked as the expected final evaluation."
        )

    return [summary_path.resolve()]


# =============================================================================
# Hugging Face access and metadata
# =============================================================================

def resolve_hf_token() -> str:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or get_token()
    )

    if not token:
        raise RuntimeError(
            "ThyroidXL is gated. Run `hf auth login` once or set HF_TOKEN."
        )

    return str(token).strip()


HF_TOKEN = resolve_hf_token()


def fetch(repo_path: str, max_attempts: int = 20) -> Path:
    repo_path = str(repo_path).replace("\\", "/").lstrip("/")

    try:
        return Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=repo_path,
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
                    repo_type="dataset",
                    filename=repo_path,
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


def patient_id_from_filename(filename: str) -> str:
    stem = Path(str(filename)).stem
    match = re.match(r"^(\d+)(?:_|$)", stem)

    if match is None:
        raise ValueError(
            f"Cannot derive patient ID from filename: {filename}"
        )

    return str(int(match.group(1)))


def annotation_frame(split: str) -> pd.DataFrame:
    path = fetch(f"{split}/{split}_annotations.json")

    with path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    rows = []

    for image in coco.get("images", []):
        filename = Path(str(image["file_name"])).name
        rows.append(
            {
                "filename": filename,
                "patient_id": patient_id_from_filename(filename),
            }
        )

    frame = (
        pd.DataFrame(rows)
        .drop_duplicates()
        .sort_values(["patient_id", "filename"])
        .reset_index(drop=True)
    )

    return frame


def original_mask_fraction(
    split: str,
    filename: str,
) -> float:
    path = fetch(f"{split}/masks/{filename}")

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise RuntimeError(
            f"Could not read expert mask: {path}"
        )

    mask = mask > 0

    if not mask.any():
        raise RuntimeError(
            f"Empty expert mask: {filename}"
        )

    return float(mask.mean())


def build_mask_fraction_table(
    split: str,
    metadata: pd.DataFrame,
    cache_path: Path,
    expected_images: int,
    expected_patients: int,
) -> pd.DataFrame:
    if cache_path.is_file():
        cached = pd.read_csv(
            cache_path,
            dtype={"patient_id": str},
        )

        if (
            len(cached) == expected_images
            and cached["filename"].nunique() == expected_images
            and cached["patient_id"].nunique() == expected_patients
            and cached["expert_mask_fraction_original"].notna().all()
        ):
            return cached

    def one(row):
        return {
            "filename": str(row.filename),
            "patient_id": str(row.patient_id),
            "expert_mask_fraction_original": original_mask_fraction(
                split,
                str(row.filename),
            ),
        }

    rows = []

    with ThreadPoolExecutor(
        max_workers=MASK_WORKERS
    ) as executor:
        futures = [
            executor.submit(one, row)
            for row in metadata.itertuples(index=False)
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"{split} expert-mask fractions",
            unit="mask",
        ):
            rows.append(future.result())

    result = (
        pd.DataFrame(rows)
        .sort_values(["patient_id", "filename"])
        .reset_index(drop=True)
    )

    if len(result) != expected_images:
        raise RuntimeError(
            f"{split} mask-fraction image count mismatch."
        )

    if result["patient_id"].nunique() != expected_patients:
        raise RuntimeError(
            f"{split} mask-fraction patient count mismatch."
        )

    result.to_csv(cache_path, index=False)

    return result


def tertiles(values):
    values = pd.Series(values, dtype=float)

    q1 = float(values.quantile(1.0 / 3.0))
    q2 = float(values.quantile(2.0 / 3.0))

    if not (0.0 < q1 < q2 < 1.0):
        raise RuntimeError(
            f"Invalid nodule-size tertiles: q1={q1}, q2={q2}"
        )

    return q1, q2


def assign_size(values, q1, q2):
    values = np.asarray(values, dtype=float)

    return np.where(
        values <= q1,
        "small",
        np.where(
            values <= q2,
            "medium",
            "large",
        ),
    )


# =============================================================================
# Prediction tables / metrics
# =============================================================================

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"patient_id": str},
    )


def prediction_column(
    frame: pd.DataFrame,
    model: str,
    level: str,
) -> str:
    if "YOLO" in str(model).upper():
        candidates = [
            "prediction_primary",
            "prediction_at_zero",
            "prediction",
            "prediction_at_selected_threshold",
            "prediction_at_frozen_threshold",
        ]
    else:
        candidates = [
            "prediction_primary",
            "prediction_at_0_5",
        ]

    for column in candidates:
        if column in frame.columns:
            return column

    raise RuntimeError(
        f"Could not find canonical {model} {level} prediction column. "
        f"Available={list(frame.columns)}"
    )


def wilson(successes, total, z=1.959963984540054):
    if total <= 0:
        return float("nan"), float("nan")

    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (
        p + z * z / (2.0 * total)
    ) / denominator
    half = (
        z
        * math.sqrt(
            p * (1.0 - p) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )

    return (
        max(0.0, centre - half),
        min(1.0, centre + half),
    )


def classification_metrics(
    labels,
    predictions,
):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    accuracy_ci = wilson(
        int((labels == predictions).sum()),
        len(labels),
    )
    sensitivity_ci = wilson(tp, tp + fn)
    specificity_ci = wilson(tn, tn + fp)

    return {
        "n": int(len(labels)),
        "benign_n": int((labels == 0).sum()),
        "malignant_n": int((labels == 1).sum()),
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "accuracy_ci_low": float(accuracy_ci[0]),
        "accuracy_ci_high": float(accuracy_ci[1]),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "sensitivity_malignant": float(
            recall_score(
                labels,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "sensitivity_ci_low": float(sensitivity_ci[0]),
        "sensitivity_ci_high": float(sensitivity_ci[1]),
        "specificity_benign": (
            float(tn / (tn + fp))
            if (tn + fp)
            else float("nan")
        ),
        "specificity_ci_low": float(specificity_ci[0]),
        "specificity_ci_high": float(specificity_ci[1]),
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


def classification_bootstrap_cis(
    frame: pd.DataFrame,
    prediction_col: str,
    seed: int,
):
    """
    Patient-aware bootstrap CIs for subgroup classification.

    The resampling unit is always patient. For image-level subgroups, all frames
    contributed by a sampled patient within that subgroup are retained, preserving
    within-patient correlation. For patient-level subgroups this reduces to a
    standard patient bootstrap.
    """
    work = frame[["patient_id", "label", prediction_col]].copy()
    if work.empty:
        return {}

    groups = {
        str(pid): grp.reset_index(drop=True)
        for pid, grp in work.groupby("patient_id", sort=False)
    }
    patient_ids = list(groups)
    rng = np.random.default_rng(int(seed))
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "sensitivity_malignant",
        "specificity_benign",
        "precision_malignant",
        "f1_malignant",
        "mcc",
    ]
    distributions = {name: [] for name in metric_names}

    for _ in range(BOOTSTRAP_SAMPLES):
        sampled_ids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        sampled = pd.concat([groups[str(pid)] for pid in sampled_ids], ignore_index=True)
        metrics = classification_metrics(sampled["label"], sampled[prediction_col])
        for name in metric_names:
            value = float(metrics.get(name, float("nan")))
            if np.isfinite(value):
                distributions[name].append(value)

    output = {
        "classification_ci_method": "patient bootstrap / patient-cluster bootstrap",
        "classification_bootstrap_samples": BOOTSTRAP_SAMPLES,
        "classification_bootstrap_patients": len(patient_ids),
    }
    for name, values in distributions.items():
        low, high = _quantile_ci(values)
        output[f"{name}_bootstrap_ci_low"] = low
        output[f"{name}_bootstrap_ci_high"] = high
        output[f"{name}_bootstrap_valid"] = int(len(values))
    return output


def subgroup_classification_rows(
    frame: pd.DataFrame,
    model: str,
    level: str,
    prediction_col: str,
):
    rows = []

    for size in ("small", "medium", "large"):
        subset = frame[
            frame["nodule_size_group"] == size
        ]

        if subset.empty:
            continue

        rows.append(
            {
                "model": model,
                "level": level,
                "nodule_size_group": size,
                **classification_metrics(
                    subset["label"],
                    subset[prediction_col],
                ),
                **classification_bootstrap_cis(
                    subset,
                    prediction_col,
                    seed=_stable_seed("classification", model, level, size),
                ),
            }
        )

    return rows


def _stable_seed(*parts) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(
        __import__("hashlib").sha256(payload).digest()[:4],
        "little",
    )
    return int((BOOTSTRAP_SEED + value) % (2**32 - 1))


def _quantile_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def segmentation_bootstrap_summary(
    frame: pd.DataFrame,
    value_col: str,
    seed: int,
):
    """
    Patient-aware uncertainty for repeated-frame segmentation data.

    image_weighted_mean: keeps the observed frame contribution per sampled patient
    and uses a patient-cluster bootstrap.

    patient_balanced_mean: averages within patient first, then gives each patient
    equal weight and bootstraps patients.
    """
    work = frame[["patient_id", value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna()

    if work.empty:
        return {
            "image_weighted_mean": float("nan"),
            "image_weighted_ci_low": float("nan"),
            "image_weighted_ci_high": float("nan"),
            "patient_balanced_mean": float("nan"),
            "patient_balanced_ci_low": float("nan"),
            "patient_balanced_ci_high": float("nan"),
            "n_patients_assessed": 0,
        }

    grouped = work.groupby("patient_id", sort=False)[value_col].agg(
        ["sum", "count", "mean"]
    )
    patient_sum = grouped["sum"].to_numpy(dtype=float)
    patient_count = grouped["count"].to_numpy(dtype=float)
    patient_mean = grouped["mean"].to_numpy(dtype=float)

    n_patients = len(grouped)
    rng = np.random.default_rng(int(seed))
    image_boot = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    patient_boot = np.empty(BOOTSTRAP_SAMPLES, dtype=float)

    for b in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, n_patients, size=n_patients)
        image_boot[b] = float(
            patient_sum[idx].sum() / patient_count[idx].sum()
        )
        patient_boot[b] = float(patient_mean[idx].mean())

    image_ci = _quantile_ci(image_boot)
    patient_ci = _quantile_ci(patient_boot)

    return {
        "image_weighted_mean": float(work[value_col].mean()),
        "image_weighted_ci_low": image_ci[0],
        "image_weighted_ci_high": image_ci[1],
        "patient_balanced_mean": float(patient_mean.mean()),
        "patient_balanced_ci_low": patient_ci[0],
        "patient_balanced_ci_high": patient_ci[1],
        "n_patients_assessed": int(n_patients),
    }


def segmentation_rows(
    frame: pd.DataFrame,
    model: str,
    dice_col: str,
    iou_col: str,
    group_column: str,
    groups,
    geometry: str,
):
    rows = []

    for group in groups:
        subset = frame[
            frame[group_column] == group
        ].copy()

        dice = pd.to_numeric(
            subset[dice_col],
            errors="coerce",
        )
        iou = pd.to_numeric(
            subset[iou_col],
            errors="coerce",
        )

        assessed = dice.notna() & iou.notna()
        assessed_frame = subset.loc[assessed].copy()
        assessed_frame[dice_col] = dice.loc[assessed].to_numpy(dtype=float)
        assessed_frame[iou_col] = iou.loc[assessed].to_numpy(dtype=float)

        dice_boot = segmentation_bootstrap_summary(
            assessed_frame,
            dice_col,
            _stable_seed(model, group_column, group, "dice"),
        )
        iou_boot = segmentation_bootstrap_summary(
            assessed_frame,
            iou_col,
            _stable_seed(model, group_column, group, "iou"),
        )

        rows.append(
            {
                "model": model,
                "group_type": group_column,
                "group": group,
                "n_images_total": int(len(subset)),
                "n_images_segmentation_assessed": int(assessed.sum()),
                "n_patients_segmentation_assessed": int(
                    dice_boot["n_patients_assessed"]
                ),
                "segmentation_assessment_coverage": (
                    float(assessed.mean())
                    if len(subset)
                    else float("nan")
                ),
                "mean_dice": (
                    float(dice[assessed].mean())
                    if assessed.any()
                    else float("nan")
                ),
                "mean_dice_patient_cluster_ci_low": dice_boot[
                    "image_weighted_ci_low"
                ],
                "mean_dice_patient_cluster_ci_high": dice_boot[
                    "image_weighted_ci_high"
                ],
                "patient_balanced_mean_dice": dice_boot[
                    "patient_balanced_mean"
                ],
                "patient_balanced_mean_dice_ci_low": dice_boot[
                    "patient_balanced_ci_low"
                ],
                "patient_balanced_mean_dice_ci_high": dice_boot[
                    "patient_balanced_ci_high"
                ],
                "median_dice": (
                    float(dice[assessed].median())
                    if assessed.any()
                    else float("nan")
                ),
                "mean_iou": (
                    float(iou[assessed].mean())
                    if assessed.any()
                    else float("nan")
                ),
                "mean_iou_patient_cluster_ci_low": iou_boot[
                    "image_weighted_ci_low"
                ],
                "mean_iou_patient_cluster_ci_high": iou_boot[
                    "image_weighted_ci_high"
                ],
                "patient_balanced_mean_iou": iou_boot[
                    "patient_balanced_mean"
                ],
                "patient_balanced_mean_iou_ci_low": iou_boot[
                    "patient_balanced_ci_low"
                ],
                "patient_balanced_mean_iou_ci_high": iou_boot[
                    "patient_balanced_ci_high"
                ],
                "median_iou": (
                    float(iou[assessed].median())
                    if assessed.any()
                    else float("nan")
                ),
                "bootstrap_samples": int(BOOTSTRAP_SAMPLES),
                "geometry": geometry,
            }
        )

    return rows





MODEL_SPECS = {
    "MobileNetV3": {
        "result_dir": "MobileNet",
        "score": "probability_malignant",
        "dice": "segmentation_dice",
        "iou": "segmentation_iou",
    },
    "EfficientNetB3": {
        "result_dir": "EfficientNetB3",
        "score": "probability_malignant",
        "dice": "segmentation_dice",
        "iou": "segmentation_iou",
    },
    "ConvNeXtTiny": {
        "result_dir": "ConvNeXtTiny",
        "score": "probability_malignant",
        "dice": "segmentation_dice",
        "iou": "segmentation_iou",
    },
    "YOLOv8sSeg": {
        "result_dir": "YOLO",
        "score": "malignant_score",
        "dice": "mask_dice",
        "iou": "mask_iou",
    },
}


def main():
    print("=" * 80)
    print("THYROIDXL FOUR-MODEL SUBGROUP ANALYSIS")
    print("=" * 80)

    model_data = {}
    for model_name, spec in MODEL_SPECS.items():
        directory = RESULTS / spec["result_dir"] / "FinalOfficialTest"
        image_path = directory / "test_image_predictions.csv"
        patient_path = directory / "test_patient_predictions.csv"
        if not image_path.is_file() or not patient_path.is_file():
            raise RuntimeError(
                f"{model_name}: missing canonical final prediction CSVs under {directory}"
            )

        provenance = verify_final_result_provenance(image_path, model_name)
        image = read_csv(image_path)
        patient = read_csv(patient_path)

        if len(image) != EXPECTED_TEST_IMAGES or image["patient_id"].nunique() != EXPECTED_TEST_PATIENTS:
            raise RuntimeError(f"{model_name}: image output count mismatch.")
        if len(patient) != EXPECTED_TEST_PATIENTS or patient["patient_id"].nunique() != EXPECTED_TEST_PATIENTS:
            raise RuntimeError(f"{model_name}: patient output count mismatch.")

        model_data[model_name] = {
            "image": image,
            "patient": patient,
            "image_path": image_path,
            "patient_path": patient_path,
            "provenance": provenance,
        }

    # Exact same held-out keys for all four models.
    reference = None
    for model_name, payload in model_data.items():
        keys = payload["image"][["filename", "patient_id", "label"]].sort_values(
            ["patient_id", "filename"]
        ).reset_index(drop=True)
        if reference is None:
            reference = keys
        elif not keys.equals(reference):
            raise RuntimeError(f"{model_name}: held-out image keys/labels do not match the other models.")

    # -----------------------------------------------------------------
    # TRAIN ONLY: derive the common size thresholds.
    # -----------------------------------------------------------------
    print("Deriving nodule-size tertiles from official TRAIN masks only...")
    train_metadata = annotation_frame("train")
    if len(train_metadata) != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError("Official training metadata image count mismatch.")
    if train_metadata["patient_id"].nunique() != EXPECTED_TRAIN_PATIENTS:
        raise RuntimeError("Official training metadata patient count mismatch.")

    train_fraction = build_mask_fraction_table(
        "train",
        train_metadata,
        OUT / "official_training_mask_fractions.csv",
        EXPECTED_TRAIN_IMAGES,
        EXPECTED_TRAIN_PATIENTS,
    )

    image_q1, image_q2 = tertiles(train_fraction["expert_mask_fraction_original"])
    train_patient_fraction = (
        train_fraction.groupby("patient_id", as_index=False)
        .agg(
            mean_expert_mask_fraction=("expert_mask_fraction_original", "mean"),
            n_frames=("filename", "size"),
        )
    )
    patient_q1, patient_q2 = tertiles(
        train_patient_fraction["mean_expert_mask_fraction"]
    )

    freeze = {
        "status": "THYROIDXL_FOUR_MODEL_COMMON_SUBGROUPS_FROZEN_FROM_OFFICIAL_TRAIN",
        "dataset_repo": REPO_ID,
        "dataset_revision": REPO_REVISION,
        "official_training_images": EXPECTED_TRAIN_IMAGES,
        "official_training_patients": EXPECTED_TRAIN_PATIENTS,
        "image_size_definition": "original-image expert nodule-mask foreground fraction",
        "image_q1": image_q1,
        "image_q2": image_q2,
        "patient_size_definition": "mean original-image expert-mask fraction across patient frames",
        "patient_q1": patient_q1,
        "patient_q2": patient_q2,
        "models": list(MODEL_SPECS),
        "official_test_used_to_define_subgroups": False,
    }
    (OUT / "common_nodule_size_freeze_4models.json").write_text(
        json.dumps(freeze, indent=2), encoding="utf-8"
    )

    # -----------------------------------------------------------------
    # TEST: apply frozen size boundaries only.
    # -----------------------------------------------------------------
    test_metadata = annotation_frame("test")
    if len(test_metadata) != EXPECTED_TEST_IMAGES:
        raise RuntimeError("Official test metadata image count mismatch.")
    if test_metadata["patient_id"].nunique() != EXPECTED_TEST_PATIENTS:
        raise RuntimeError("Official test metadata patient count mismatch.")

    test_fraction = build_mask_fraction_table(
        "test",
        test_metadata,
        OUT / "official_test_mask_fractions.csv",
        EXPECTED_TEST_IMAGES,
        EXPECTED_TEST_PATIENTS,
    )
    test_fraction["nodule_size_group"] = assign_size(
        test_fraction["expert_mask_fraction_original"], image_q1, image_q2
    )
    test_patient_fraction = (
        test_fraction.groupby("patient_id", as_index=False)
        .agg(
            mean_expert_mask_fraction=("expert_mask_fraction_original", "mean"),
            n_frames=("filename", "size"),
        )
    )
    test_patient_fraction["nodule_size_group"] = assign_size(
        test_patient_fraction["mean_expert_mask_fraction"], patient_q1, patient_q2
    )

    classification_rows = []
    segmentation_size_rows = []
    segmentation_diagnosis_rows = []
    segmentation_correctness_rows = []
    merged_cases = None

    for model_name, spec in MODEL_SPECS.items():
        image = model_data[model_name]["image"].merge(
            test_fraction,
            on=["filename", "patient_id"],
            how="inner",
            validate="one_to_one",
        )
        patient = model_data[model_name]["patient"].merge(
            test_patient_fraction[
                ["patient_id", "mean_expert_mask_fraction", "nodule_size_group"]
            ],
            on="patient_id",
            how="inner",
            validate="one_to_one",
        )

        image_pred = prediction_column(image, model_name, "image")
        patient_pred = prediction_column(patient, model_name, "patient")

        classification_rows.extend(
            subgroup_classification_rows(image, model_name, "image", image_pred)
        )
        classification_rows.extend(
            subgroup_classification_rows(patient, model_name, "patient", patient_pred)
        )

        required_seg = {spec["dice"], spec["iou"], "segmentation_geometry"}
        missing = required_seg - set(image.columns)
        if missing:
            raise RuntimeError(f"{model_name}: missing segmentation columns {sorted(missing)}")
        geometry = set(image["segmentation_geometry"].astype(str).str.lower())
        if geometry != {"original_ultrasound"}:
            raise RuntimeError(
                f"{model_name}: segmentation geometry is not original_ultrasound: {geometry}"
            )

        image["diagnosis_group"] = image["label"].map({0: "benign", 1: "malignant"})
        image["correctness_group"] = np.where(
            image[image_pred].astype(int) == image["label"].astype(int),
            "correct",
            "incorrect",
        )

        segmentation_size_rows.extend(
            segmentation_rows(
                image, model_name, spec["dice"], spec["iou"],
                "nodule_size_group", ("small", "medium", "large"),
                "original_ultrasound",
            )
        )
        segmentation_diagnosis_rows.extend(
            segmentation_rows(
                image, model_name, spec["dice"], spec["iou"],
                "diagnosis_group", ("benign", "malignant"),
                "original_ultrasound",
            )
        )
        segmentation_correctness_rows.extend(
            segmentation_rows(
                image, model_name, spec["dice"], spec["iou"],
                "correctness_group", ("correct", "incorrect"),
                "original_ultrasound",
            )
        )

        case = image[
            [
                "filename", "patient_id", "label",
                "expert_mask_fraction_original", "nodule_size_group",
                spec["score"], image_pred, spec["dice"], spec["iou"],
            ]
        ].copy()
        case = case.rename(columns={
            spec["score"]: f"{model_name}__score",
            image_pred: f"{model_name}__prediction",
            spec["dice"]: f"{model_name}__dice",
            spec["iou"]: f"{model_name}__iou",
        })
        if merged_cases is None:
            merged_cases = case
        else:
            # Avoid duplicate common size columns on the later merges.
            case = case.drop(
                columns=["expert_mask_fraction_original", "nodule_size_group"]
            )
            merged_cases = merged_cases.merge(
                case,
                on=["filename", "patient_id", "label"],
                how="inner",
                validate="one_to_one",
            )

    pd.DataFrame(classification_rows).to_csv(
        OUT / "classification_by_common_nodule_size_4models.csv", index=False
    )
    pd.DataFrame(segmentation_size_rows).to_csv(
        OUT / "segmentation_by_common_nodule_size_4models.csv", index=False
    )
    pd.DataFrame(segmentation_diagnosis_rows).to_csv(
        OUT / "segmentation_by_diagnosis_4models.csv", index=False
    )
    pd.DataFrame(segmentation_correctness_rows).to_csv(
        OUT / "segmentation_by_classification_correctness_4models.csv", index=False
    )
    merged_cases.to_csv(
        OUT / "four_model_case_table_with_common_size.csv", index=False
    )

    manifest = {
        "status": "THYROIDXL_FOUR_MODEL_SUBGROUP_ANALYSIS_COMPLETE",
        "dataset_revision": REPO_REVISION,
        "models": list(MODEL_SPECS),
        "official_test_images": EXPECTED_TEST_IMAGES,
        "official_test_patients": EXPECTED_TEST_PATIENTS,
        "size_freeze": str(OUT / "common_nodule_size_freeze_4models.json"),
        "segmentation_geometry": "original_ultrasound",
        "official_test_used_to_define_subgroups": False,
        "provenance": {
            name: [str(p) for p in payload["provenance"]]
            for name, payload in model_data.items()
        },
    }
    (OUT / "subgroup_analysis_manifest_4models.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Four-model subgroup analysis complete.")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
