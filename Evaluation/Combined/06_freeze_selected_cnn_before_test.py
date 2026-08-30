from __future__ import annotations

"""
ThyroidXL — prospective selected-CNN freeze for the dual-component analysis.

PURPOSE
-------
Choose the ONE CNN that will form the dedicated CNN↔YOLO dual-component
analysis BEFORE any official held-out predictions are used for model selection.

Frozen rule:
    select the CNN with the highest Fold-1 DEVELOPMENT patient ROC-AUC
    recorded in the FINAL full-training checkpoint's selection_source.

Models considered:
- MobileNetV3-Large
- EfficientNet-B3
- ConvNeXt-Tiny

SAFETY
------
- This script NEVER downloads or opens official test metadata/images/masks.
- It verifies that each candidate is a FINAL model trained on all
  9,541 official-training images / 3,354 patients.
- It verifies that the selection source is the patient-disjoint Fold 1:
  7,684 train images / 2,683 patients;
  1,857 validation images / 671 patients;
  zero patient overlap.
- It requires checkpoint selection by validation patient ROC-AUC.
- Exact ties stop safely instead of inventing a post-hoc tie-breaker.
"""

import hashlib
import json
import os
from pathlib import Path

import torch


EXPECTED_TRAIN_IMAGES = 9541
EXPECTED_TRAIN_PATIENTS = 3354
EXPECTED_FOLD1_TRAIN_IMAGES = 7684
EXPECTED_FOLD1_TRAIN_PATIENTS = 2683
EXPECTED_FOLD1_VAL_IMAGES = 1857
EXPECTED_FOLD1_VAL_PATIENTS = 671

MODEL_SPECS = {
    "MobileNetV3": {
        "directory": "MobileNetV3",
        "status": (
            "THYROIDXL_MOBILENET_FINAL_OFFICIAL_TRAIN_"
            "REFIT_ONEFOLD_SELECTED"
        ),
    },
    "EfficientNetB3": {
        "directory": "EfficientNetB3",
        "status": (
            "THYROIDXL_EFFICIENTNETB3_FINAL_OFFICIAL_TRAIN_"
            "REFIT_ONEFOLD_SELECTED"
        ),
    },
    "ConvNeXtTiny": {
        "directory": "ConvNeXtTiny",
        "status": (
            "THYROIDXL_CONVNEXTTINY_FINAL_OFFICIAL_TRAIN_"
            "REFIT_ONEFOLD_SELECTED"
        ),
    },
}

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root() -> Path:
    env = os.environ.get("THYROIDXL_PROJECT_ROOT")
    if env:
        candidate = Path(env).expanduser().resolve()
        if candidate.is_dir():
            return candidate

    for candidate in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (candidate / "Models").is_dir():
            return candidate

    raise RuntimeError(
        "Could not locate ThyroidXL project root. Expected Models/ above "
        "this script, or set THYROIDXL_PROJECT_ROOT."
    )


ROOT = find_project_root()
OUT = ROOT / "results" / "Combined"
OUT.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_load_compatible(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def final_checkpoint(model_name: str, spec: dict):
    model_dir = ROOT / "Models" / spec["directory"]
    if not model_dir.is_dir():
        raise FileNotFoundError(f"{model_name}: missing model directory {model_dir}")

    matches = []
    for path in model_dir.rglob("*.pt"):
        try:
            payload = torch_load_compatible(path)
        except Exception:
            continue
        if (
            isinstance(payload, dict)
            and payload.get("status") == spec["status"]
        ):
            matches.append((path.resolve(), payload))

    if len(matches) != 1:
        raise RuntimeError(
            f"{model_name}: expected exactly one FINAL checkpoint with "
            f"status={spec['status']!r}; found {len(matches)}:\n"
            + "\n".join(str(path) for path, _ in matches)
        )

    path, payload = matches[0]

    if payload.get("official_test_accessed") is not False:
        raise RuntimeError(
            f"{model_name}: final checkpoint does not certify untouched test."
        )
    if int(payload.get("training_images", -1)) != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError(f"{model_name}: final training-image count mismatch.")
    if int(payload.get("training_patients", -1)) != EXPECTED_TRAIN_PATIENTS:
        raise RuntimeError(f"{model_name}: final training-patient count mismatch.")

    selection = payload.get("selection_source")
    if not isinstance(selection, dict):
        raise RuntimeError(f"{model_name}: missing selection_source metadata.")

    expected = {
        "training_images": EXPECTED_FOLD1_TRAIN_IMAGES,
        "training_patients": EXPECTED_FOLD1_TRAIN_PATIENTS,
        "validation_images": EXPECTED_FOLD1_VAL_IMAGES,
        "validation_patients": EXPECTED_FOLD1_VAL_PATIENTS,
        "patient_overlap": 0,
    }
    for key, value in expected.items():
        if int(selection.get(key, -1)) != value:
            raise RuntimeError(
                f"{model_name}: selection_source {key} mismatch. "
                f"Expected {value}, got {selection.get(key)!r}."
            )

    metric = str(selection.get("checkpoint_selection_metric", "")).lower()
    if not (
        "patient" in metric
        and "roc" in metric
        and "auc" in metric
    ):
        raise RuntimeError(
            f"{model_name}: checkpoint selection metric is not patient ROC-AUC: "
            f"{metric!r}"
        )

    auc = float(selection.get("best_validation_patient_auc", float("nan")))
    if not (0.0 <= auc <= 1.0):
        raise RuntimeError(
            f"{model_name}: invalid development patient ROC-AUC {auc!r}."
        )

    return {
        "model": model_name,
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "development_patient_roc_auc": auc,
        "best_phase": selection.get("best_phase"),
        "best_epoch": int(selection.get("best_epoch", -1)),
        "training_status": payload.get("status"),
    }


def main():
    candidates = [
        final_checkpoint(name, spec)
        for name, spec in MODEL_SPECS.items()
    ]
    candidates.sort(
        key=lambda row: row["development_patient_roc_auc"],
        reverse=True,
    )

    if len(candidates) >= 2:
        gap = (
            candidates[0]["development_patient_roc_auc"]
            - candidates[1]["development_patient_roc_auc"]
        )
        if abs(gap) <= 1e-12:
            raise RuntimeError(
                "The top CNNs are exactly tied under the prospectively frozen "
                "development patient ROC-AUC rule. Define a tie-breaker before "
                "opening the official test."
            )

    selected = candidates[0]

    freeze = {
        "status": "THYROIDXL_SELECTED_CNN_FROZEN_BEFORE_OFFICIAL_TEST",
        "selection_rule": (
            "highest Fold-1 development patient ROC-AUC recorded in each "
            "final full-training checkpoint"
        ),
        "official_test_accessed": False,
        "official_test_used_for_selection": False,
        "selected_cnn": selected["model"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_development_patient_roc_auc": (
            selected["development_patient_roc_auc"]
        ),
        "all_candidates": candidates,
    }

    freeze_path = OUT / "selected_cnn_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("THYROIDXL PROSPECTIVE SELECTED-CNN FREEZE")
    print("=" * 80)
    for row in candidates:
        print(
            f"{row['model']}: development patient ROC-AUC "
            f"{row['development_patient_roc_auc']:.6f}"
        )
    print()
    print("SELECTED CNN:", selected["model"])
    print("Freeze artifact:", freeze_path)
    print("Official test accessed by this script: NO")


if __name__ == "__main__":
    main()
