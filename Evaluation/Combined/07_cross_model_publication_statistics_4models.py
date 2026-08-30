"""
ThyroidXL — final four-model publication statistics.

Compares the FINAL held-out predictions from:
- MobileNetV3-Large
- EfficientNet-B3
- ConvNeXt-Tiny
- YOLOv8s-seg

All four must contain the same official held-out cohort:
2,094 images / 739 patients.

Scientific rules
----------------
- CNN ranking scores are malignant probabilities.
- YOLO ranking score is signed detector evidence and is NOT treated as a
  calibrated probability.
- Image-level uncertainty uses patient-cluster bootstrap.
- Patient-level uncertainty uses paired patient bootstrap.
- All pairwise comparisons use the same resampled units within each iteration.
- Patient-level paired correctness uses exact McNemar tests.
- McNemar p-values are Holm-adjusted across the six pairwise model tests.
- Segmentation pairwise comparisons require original-ultrasound geometry.
- Wilcoxon p-values are Holm-adjusted across the six model pairs separately
  for Dice and IoU.
- The dedicated CNN↔YOLO pair is read from a development-only freeze created
  before official-test evaluation; it is never selected from test performance.
- No test-set threshold/model/epoch tuning is performed here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
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

EXPECTED_TRAIN_IMAGES = 9541
EXPECTED_TRAIN_PATIENTS = 3354
EXPECTED_TEST_IMAGES = 2094
EXPECTED_TEST_PATIENTS = 739
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_SPECS = {
    "MobileNetV3": {
        "result_dir": "MobileNet",
        "score_col": "probability_malignant",
        "pred_candidates": ["prediction_primary", "prediction_at_0_5"],
        "seg_dice": "segmentation_dice",
        "seg_iou": "segmentation_iou",
        "status_token": "MOBILENET",
        "probability": True,
    },
    "EfficientNetB3": {
        "result_dir": "EfficientNetB3",
        "score_col": "probability_malignant",
        "pred_candidates": ["prediction_primary", "prediction_at_0_5"],
        "seg_dice": "segmentation_dice",
        "seg_iou": "segmentation_iou",
        "status_token": "EFFICIENTNETB3",
        "probability": True,
    },
    "ConvNeXtTiny": {
        "result_dir": "ConvNeXtTiny",
        "score_col": "probability_malignant",
        "pred_candidates": ["prediction_primary", "prediction_at_0_5"],
        "seg_dice": "segmentation_dice",
        "seg_iou": "segmentation_iou",
        "status_token": "CONVNEXTTINY",
        "probability": True,
    },
    "YOLOv8sSeg": {
        "result_dir": "YOLO",
        "score_col": "malignant_score",
        "pred_candidates": ["prediction_primary", "prediction_at_zero"],
        "seg_dice": "mask_dice",
        "seg_iou": "mask_iou",
        "status_token": "YOLO",
        "probability": False,
    },
}


def find_project_root() -> Path:
    env = os.environ.get("THYROIDXL_PROJECT_ROOT")
    if env:
        candidate = Path(env).expanduser().resolve()
        if candidate.is_dir():
            return candidate
    for candidate in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (candidate / "Models").is_dir() and (candidate / "results").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate ThyroidXL root. Expected Models/ and results/ above "
        "this script, or set THYROIDXL_PROJECT_ROOT."
    )


ROOT = find_project_root()
RESULTS = ROOT / "results"
OUT = RESULTS / "Combined" / "Statistics4Models" / "FinalOfficialTest"
OUT.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holm_adjust(p_values):
    """Holm step-down family-wise-error correction."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)

    finite_idx = np.flatnonzero(np.isfinite(values))
    if finite_idx.size == 0:
        return adjusted

    order = finite_idx[np.argsort(values[finite_idx])]
    m = len(order)
    running = 0.0

    for rank, idx in enumerate(order):
        raw = float(values[idx])
        candidate = (m - rank) * raw
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)

    return adjusted


def load_selected_cnn_freeze():
    path = RESULTS / "Combined" / "selected_cnn_freeze.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Missing results/Combined/selected_cnn_freeze.json. Run "
            "06_freeze_selected_cnn_before_test.py BEFORE official-test "
            "evaluation so the dedicated CNN↔YOLO pair is prospectively frozen."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "THYROIDXL_SELECTED_CNN_FROZEN_BEFORE_OFFICIAL_TEST":
        raise RuntimeError("Selected-CNN freeze has an unexpected status.")
    if payload.get("official_test_accessed") is not False:
        raise RuntimeError(
            "Selected-CNN freeze does not certify official_test_accessed=false."
        )
    if payload.get("official_test_used_for_selection") is not False:
        raise RuntimeError(
            "Selected-CNN freeze does not certify test-independent selection."
        )

    selected = str(payload.get("selected_cnn", "")).strip()
    allowed = {"MobileNetV3", "EfficientNetB3", "ConvNeXtTiny"}
    if selected not in allowed:
        raise RuntimeError(
            f"Selected-CNN freeze contains invalid selected_cnn={selected!r}."
        )

    sha = str(payload.get("selected_checkpoint_sha256", "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise RuntimeError("Selected-CNN freeze has no valid checkpoint SHA256.")

    return path.resolve(), payload


def is_primary_dual_pair(a: str, b: str, selected_cnn: str) -> bool:
    return {str(a), str(b)} == {str(selected_cnn), "YOLOv8sSeg"}


def metric_bundle(labels, scores, predictions):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }


METRICS = (
    "roc_auc", "auprc", "accuracy", "balanced_accuracy",
    "sensitivity", "specificity", "precision", "f1", "mcc",
)


def choose_prediction_column(frame: pd.DataFrame, candidates):
    for column in candidates:
        if column in frame.columns:
            return column
    raise RuntimeError(
        f"No canonical prediction column. Candidates={candidates}; "
        f"available={list(frame.columns)}"
    )


def validate_summary(model_name: str, directory: Path):
    summary_path = directory / "test_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"{model_name}: missing {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(payload.get("official_training_images", -1)) != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError(f"{model_name}: summary does not certify 9,541 train images.")
    if int(payload.get("official_training_patients", -1)) != EXPECTED_TRAIN_PATIENTS:
        raise RuntimeError(f"{model_name}: summary does not certify 3,354 train patients.")
    if int(payload.get("official_test_images", -1)) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(f"{model_name}: summary does not certify 2,094 test images.")
    if int(payload.get("official_test_patients", -1)) != EXPECTED_TEST_PATIENTS:
        raise RuntimeError(f"{model_name}: summary does not certify 739 test patients.")
    if payload.get("official_test_used_for_tuning") is not False:
        raise RuntimeError(f"{model_name}: test tuning is not explicitly false.")
    sha = str(payload.get("checkpoint_sha256", "")).lower().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise RuntimeError(f"{model_name}: invalid checkpoint SHA in test summary.")
    status = str(payload.get("status", "")).upper()
    token = MODEL_SPECS[model_name]["status_token"]
    if token not in status or "FINAL" not in status:
        raise RuntimeError(f"{model_name}: summary status is not final: {status!r}")
    return summary_path, payload


def load_model_outputs(model_name: str):
    spec = MODEL_SPECS[model_name]
    directory = RESULTS / spec["result_dir"] / "FinalOfficialTest"
    image_path = directory / "test_image_predictions.csv"
    patient_path = directory / "test_patient_predictions.csv"
    if not image_path.is_file() or not patient_path.is_file():
        raise RuntimeError(
            f"{model_name}: missing canonical prediction CSVs under {directory}"
        )
    summary_path, summary = validate_summary(model_name, directory)
    image = pd.read_csv(image_path, dtype={"patient_id": str})
    patient = pd.read_csv(patient_path, dtype={"patient_id": str})
    if len(image) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(f"{model_name}: image prediction row count != 2,094")
    if len(patient) != EXPECTED_TEST_PATIENTS:
        raise RuntimeError(f"{model_name}: patient prediction row count != 739")
    if image["patient_id"].nunique() != EXPECTED_TEST_PATIENTS:
        raise RuntimeError(f"{model_name}: image CSV does not contain 739 patients")
    if image["filename"].duplicated().any():
        raise RuntimeError(f"{model_name}: duplicate image filenames")
    pred_image = choose_prediction_column(image, spec["pred_candidates"])
    pred_patient = choose_prediction_column(patient, spec["pred_candidates"])
    score = spec["score_col"]
    for frame, level, pred in [(image, "image", pred_image), (patient, "patient", pred_patient)]:
        for column in ("label", score, pred):
            if column not in frame.columns:
                raise RuntimeError(f"{model_name}: missing {level} column {column!r}")
    return {
        "image": image,
        "patient": patient,
        "image_pred": pred_image,
        "patient_pred": pred_patient,
        "summary_path": summary_path,
        "summary": summary,
    }


def align_level(outputs, level: str):
    key_cols = ["patient_id", "label"] if level == "patient" else ["filename", "patient_id", "label"]
    merged = None
    for model_name, payload in outputs.items():
        spec = MODEL_SPECS[model_name]
        frame = payload[level].copy()
        pred_col = payload[f"{level}_pred"]
        subset = frame[key_cols + [spec["score_col"], pred_col]].copy()
        subset = subset.rename(columns={
            spec["score_col"]: f"{model_name}__score",
            pred_col: f"{model_name}__prediction",
        })
        merged = subset if merged is None else merged.merge(
            subset, on=key_cols, how="inner", validate="one_to_one"
        )
    expected = EXPECTED_TEST_PATIENTS if level == "patient" else EXPECTED_TEST_IMAGES
    if len(merged) != expected:
        raise RuntimeError(
            f"Four-model {level} alignment produced {len(merged)} rows, expected {expected}."
        )
    return merged


def diagnostic_table(aligned: pd.DataFrame, level: str):
    rows = []
    for model_name in MODEL_SPECS:
        metrics = metric_bundle(
            aligned["label"],
            aligned[f"{model_name}__score"],
            aligned[f"{model_name}__prediction"],
        )
        for metric, value in metrics.items():
            rows.append({
                "level": level,
                "model": model_name,
                "metric": metric,
                "value": value,
            })
    return pd.DataFrame(rows)


def paired_patient_bootstrap(patient: pd.DataFrame):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    observed = {}
    for model_name in MODEL_SPECS:
        observed[model_name] = metric_bundle(
            patient["label"],
            patient[f"{model_name}__score"],
            patient[f"{model_name}__prediction"],
        )

    samples = {pair: {m: [] for m in METRICS}
               for pair in combinations(MODEL_SPECS, 2)}
    n = len(patient)
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, n, n)
        boot = patient.iloc[idx]
        if boot["label"].nunique() < 2:
            continue
        metrics = {
            model: metric_bundle(
                boot["label"],
                boot[f"{model}__score"],
                boot[f"{model}__prediction"],
            )
            for model in MODEL_SPECS
        }
        for a, b in combinations(MODEL_SPECS, 2):
            for metric in METRICS:
                samples[(a, b)][metric].append(metrics[a][metric] - metrics[b][metric])

    rows = []
    for a, b in combinations(MODEL_SPECS, 2):
        for metric in METRICS:
            vals = np.asarray(samples[(a, b)][metric], dtype=float)
            vals = vals[np.isfinite(vals)]
            rows.append({
                "level": "patient",
                "model_a": a,
                "model_b": b,
                "metric": metric,
                "difference_a_minus_b": observed[a][metric] - observed[b][metric],
                "ci_low": float(np.quantile(vals, 0.025)) if len(vals) else np.nan,
                "ci_high": float(np.quantile(vals, 0.975)) if len(vals) else np.nan,
                "bootstrap_samples_valid": int(len(vals)),
            })
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(image: pd.DataFrame):
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    patient_ids = image["patient_id"].drop_duplicates().to_numpy()
    observed = {
        model: metric_bundle(
            image["label"],
            image[f"{model}__score"],
            image[f"{model}__prediction"],
        )
        for model in MODEL_SPECS
    }
    samples = {pair: {m: [] for m in METRICS}
               for pair in combinations(MODEL_SPECS, 2)}
    grouped = {pid: image[image["patient_id"] == pid] for pid in patient_ids}
    for _ in range(BOOTSTRAP_SAMPLES):
        chosen = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        boot_parts = []
        for j, pid in enumerate(chosen):
            part = grouped[pid].copy()
            part["_bootstrap_cluster"] = j
            boot_parts.append(part)
        boot = pd.concat(boot_parts, ignore_index=True)
        if boot["label"].nunique() < 2:
            continue
        metrics = {
            model: metric_bundle(
                boot["label"],
                boot[f"{model}__score"],
                boot[f"{model}__prediction"],
            )
            for model in MODEL_SPECS
        }
        for a, b in combinations(MODEL_SPECS, 2):
            for metric in METRICS:
                samples[(a, b)][metric].append(metrics[a][metric] - metrics[b][metric])
    rows = []
    for a, b in combinations(MODEL_SPECS, 2):
        for metric in METRICS:
            vals = np.asarray(samples[(a, b)][metric], dtype=float)
            vals = vals[np.isfinite(vals)]
            rows.append({
                "level": "image_patient_cluster",
                "model_a": a,
                "model_b": b,
                "metric": metric,
                "difference_a_minus_b": observed[a][metric] - observed[b][metric],
                "ci_low": float(np.quantile(vals, 0.025)) if len(vals) else np.nan,
                "ci_high": float(np.quantile(vals, 0.975)) if len(vals) else np.nan,
                "bootstrap_samples_valid": int(len(vals)),
            })
    return pd.DataFrame(rows)


def exact_mcnemar_all_pairs(patient: pd.DataFrame):
    labels = patient["label"].to_numpy(dtype=int)
    rows = []
    for a, b in combinations(MODEL_SPECS, 2):
        ca = patient[f"{a}__prediction"].to_numpy(dtype=int) == labels
        cb = patient[f"{b}__prediction"].to_numpy(dtype=int) == labels
        a_only = int(np.sum(ca & ~cb))
        b_only = int(np.sum(~ca & cb))
        discordant = a_only + b_only
        p = 1.0 if discordant == 0 else float(
            binomtest(min(a_only, b_only), discordant, 0.5, alternative="two-sided").pvalue
        )
        rows.append({
            "model_a": a,
            "model_b": b,
            "a_correct_b_wrong": a_only,
            "a_wrong_b_correct": b_only,
            "discordant": discordant,
            "exact_mcnemar_p": p,
        })
    result = pd.DataFrame(rows)
    result["exact_mcnemar_p_holm"] = holm_adjust(
        result["exact_mcnemar_p"].to_numpy(dtype=float)
    )
    result["reject_holm_0_05"] = (
        result["exact_mcnemar_p_holm"] < 0.05
    )
    return result


def segmentation_tables(outputs):
    merged = None
    key = ["filename", "patient_id", "label"]
    for model_name, payload in outputs.items():
        spec = MODEL_SPECS[model_name]
        frame = payload["image"]
        required = {spec["seg_dice"], spec["seg_iou"], "segmentation_geometry"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{model_name}: missing segmentation columns {sorted(missing)}")
        geometry = set(frame["segmentation_geometry"].astype(str).str.lower())
        if geometry != {"original_ultrasound"}:
            raise RuntimeError(
                f"{model_name}: segmentation geometry must be original_ultrasound; got {geometry}"
            )
        subset = frame[key + [spec["seg_dice"], spec["seg_iou"]]].rename(columns={
            spec["seg_dice"]: f"{model_name}__dice",
            spec["seg_iou"]: f"{model_name}__iou",
        })
        merged = subset if merged is None else merged.merge(
            subset, on=key, how="inner", validate="one_to_one"
        )
    if len(merged) != EXPECTED_TEST_IMAGES:
        raise RuntimeError("Four-model segmentation alignment is incomplete.")

    descriptive = []
    for model_name in MODEL_SPECS:
        for metric in ("dice", "iou"):
            values = merged[f"{model_name}__{metric}"].to_numpy(dtype=float)
            descriptive.append({
                "model": model_name,
                "metric": metric,
                "n_images": int(np.isfinite(values).sum()),
                "mean": float(np.nanmean(values)),
                "median": float(np.nanmedian(values)),
            })

    patient_means = merged.groupby("patient_id", as_index=False).agg(
        label=("label", "first"),
        **{
            f"{model}__{metric}": (f"{model}__{metric}", "mean")
            for model in MODEL_SPECS
            for metric in ("dice", "iou")
        }
    )

    pair_rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 2)
    patient_ids = merged["patient_id"].drop_duplicates().to_numpy()
    grouped = {pid: merged[merged["patient_id"] == pid] for pid in patient_ids}
    for a, b in combinations(MODEL_SPECS, 2):
        for metric in ("dice", "iou"):
            col_a, col_b = f"{a}__{metric}", f"{b}__{metric}"
            observed = float(np.nanmean(merged[col_a] - merged[col_b]))
            bootdiff = []
            for _ in range(BOOTSTRAP_SAMPLES):
                chosen = rng.choice(patient_ids, size=len(patient_ids), replace=True)
                part = pd.concat([grouped[pid] for pid in chosen], ignore_index=True)
                bootdiff.append(float(np.nanmean(part[col_a] - part[col_b])))
            vals = np.asarray(bootdiff, dtype=float)
            vals = vals[np.isfinite(vals)]
            pa = patient_means[col_a].to_numpy(dtype=float)
            pb = patient_means[col_b].to_numpy(dtype=float)
            mask = np.isfinite(pa) & np.isfinite(pb)
            try:
                w = wilcoxon(pa[mask], pb[mask], alternative="two-sided", zero_method="wilcox")
                w_stat, w_p = float(w.statistic), float(w.pvalue)
            except ValueError:
                w_stat, w_p = np.nan, 1.0
            pair_rows.append({
                "model_a": a,
                "model_b": b,
                "metric": metric,
                "mean_image_difference_a_minus_b": observed,
                "cluster_bootstrap_ci_low": float(np.quantile(vals, 0.025)),
                "cluster_bootstrap_ci_high": float(np.quantile(vals, 0.975)),
                "patient_wilcoxon_statistic": w_stat,
                "patient_wilcoxon_p": w_p,
            })
    pair_df = pd.DataFrame(pair_rows)
    pair_df["patient_wilcoxon_p_holm"] = np.nan
    pair_df["reject_holm_0_05"] = False

    # Multiplicity families are defined prospectively by segmentation metric:
    # six model-pair Wilcoxon tests for Dice, and six for IoU.
    for metric in ("dice", "iou"):
        mask = pair_df["metric"] == metric
        adjusted = holm_adjust(
            pair_df.loc[mask, "patient_wilcoxon_p"].to_numpy(dtype=float)
        )
        pair_df.loc[mask, "patient_wilcoxon_p_holm"] = adjusted
        pair_df.loc[mask, "reject_holm_0_05"] = adjusted < 0.05

    return merged, pd.DataFrame(descriptive), pair_df


def plot_curves(patient):
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name in MODEL_SPECS:
        RocCurveDisplay.from_predictions(
            patient["label"],
            patient[f"{model_name}__score"],
            name=model_name,
            ax=ax,
        )
    ax.set_title("ThyroidXL final patient-level ROC curves")
    fig.tight_layout()
    fig.savefig(OUT / "patient_roc_curves_4models.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name in MODEL_SPECS:
        PrecisionRecallDisplay.from_predictions(
            patient["label"],
            patient[f"{model_name}__score"],
            name=model_name,
            ax=ax,
        )
    ax.set_title("ThyroidXL final patient-level precision-recall curves")
    fig.tight_layout()
    fig.savefig(OUT / "patient_precision_recall_curves_4models.png", dpi=300)
    plt.close(fig)


def main():
    freeze_path, freeze = load_selected_cnn_freeze()
    selected_cnn = freeze["selected_cnn"]

    outputs = {name: load_model_outputs(name) for name in MODEL_SPECS}

    # Publication provenance: the prospectively selected CNN must be the exact
    # checkpoint later evaluated on the official held-out cohort.
    frozen_sha = str(
        freeze["selected_checkpoint_sha256"]
    ).strip().lower()
    evaluated_sha = str(
        outputs[selected_cnn]["summary"]["checkpoint_sha256"]
    ).strip().lower()
    if evaluated_sha != frozen_sha:
        raise RuntimeError(
            "Prospective selected-CNN checkpoint SHA does not match the "
            "checkpoint evaluated on the official held-out cohort: "
            f"selected_cnn={selected_cnn}, frozen_sha={frozen_sha}, "
            f"evaluated_sha={evaluated_sha}"
        )

    image = align_level(outputs, "image")
    patient = align_level(outputs, "patient")

    diagnostic = pd.concat(
        [diagnostic_table(image, "image"), diagnostic_table(patient, "patient")],
        ignore_index=True,
    )
    diagnostic.to_csv(OUT / "diagnostic_metrics_4models.csv", index=False)

    patient_boot = paired_patient_bootstrap(patient)
    image_boot = paired_cluster_bootstrap(image)
    paired_boot = pd.concat([patient_boot, image_boot], ignore_index=True)
    paired_boot["primary_dual_component_pair"] = [
        is_primary_dual_pair(a, b, selected_cnn)
        for a, b in zip(paired_boot["model_a"], paired_boot["model_b"])
    ]
    paired_boot.to_csv(OUT / "paired_bootstrap_differences_4models.csv", index=False)

    mcnemar = exact_mcnemar_all_pairs(patient)
    mcnemar["primary_dual_component_pair"] = [
        is_primary_dual_pair(a, b, selected_cnn)
        for a, b in zip(mcnemar["model_a"], mcnemar["model_b"])
    ]
    mcnemar.to_csv(OUT / "patient_exact_mcnemar_4models.csv", index=False)

    seg_merged, seg_desc, seg_pairs = segmentation_tables(outputs)
    seg_pairs["primary_dual_component_pair"] = [
        is_primary_dual_pair(a, b, selected_cnn)
        for a, b in zip(seg_pairs["model_a"], seg_pairs["model_b"])
    ]
    seg_desc.to_csv(OUT / "segmentation_descriptive_4models.csv", index=False)
    seg_pairs.to_csv(OUT / "segmentation_pairwise_4models.csv", index=False)

    image.to_csv(OUT / "aligned_image_predictions_4models.csv", index=False)
    patient.to_csv(OUT / "aligned_patient_predictions_4models.csv", index=False)
    plot_curves(patient)

    manifest = {
        "status": "THYROIDXL_FOUR_MODEL_FINAL_STATISTICAL_ANALYSIS",
        "official_test_images": EXPECTED_TEST_IMAGES,
        "official_test_patients": EXPECTED_TEST_PATIENTS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "models": list(MODEL_SPECS),
        "cnn_scores_are_probabilities": [
            name for name, spec in MODEL_SPECS.items() if spec["probability"]
        ],
        "nonprobability_scores": {
            "YOLOv8sSeg": (
                "max malignant confidence - max benign confidence; "
                "used for ranking/discrimination only"
            )
        },
        "pairwise_comparisons": [
            [a, b] for a, b in combinations(MODEL_SPECS, 2)
        ],
        "prospectively_selected_dual_component_cnn": selected_cnn,
        "selected_cnn_freeze": str(freeze_path),
        "selected_cnn_freeze_checkpoint_sha256": (
            freeze["selected_checkpoint_sha256"]
        ),
        "primary_dual_component_pair": [
            selected_cnn,
            "YOLOv8sSeg",
        ],
        "multiplicity_policy": {
            "patient_exact_mcnemar": (
                "Holm correction across all six pairwise model comparisons"
            ),
            "segmentation_patient_wilcoxon": (
                "Holm correction across six pairwise comparisons separately "
                "within Dice and IoU families"
            ),
            "bootstrap_effect_estimates": (
                "effect sizes and 95% paired bootstrap confidence intervals; "
                "no p-value multiplicity adjustment applicable"
            ),
        },
        "sources": {
            name: {
                "summary": str(payload["summary_path"]),
                "checkpoint_sha256": payload["summary"]["checkpoint_sha256"],
            }
            for name, payload in outputs.items()
        },
        "official_test_used_for_tuning": False,
    }
    (OUT / "statistical_analysis_manifest_4models.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Four-model publication statistics complete.")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
