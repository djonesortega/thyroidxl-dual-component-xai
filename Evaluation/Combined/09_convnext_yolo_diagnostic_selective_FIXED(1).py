from __future__ import annotations

"""
ThyroidXL — primary ConvNeXt-Tiny ↔ YOLOv8s-seg diagnostic/selective analysis.

Purpose
-------
This script evaluates the prospectively frozen diagnostic pair:
    ConvNeXt-Tiny + YOLOv8s-seg

It NEVER selects the CNN from held-out performance. The selected CNN must already
be frozen in:
    results/Combined/selected_cnn_freeze.json
and that file must certify official_test_accessed=false and
official_test_used_for_selection=false.

Primary diagnostic rule
-----------------------
ConvNeXt-Tiny:
    patient prediction from the FINAL evaluator's primary fixed 0.5 rule.
YOLOv8s-seg:
    patient prediction from the FINAL evaluator's primary signed-score > 0 rule.
Consensus:
    retain a patient only when the two primary predictions agree.

Secondary pre-specified sensitivity analysis
---------------------------------------------
If the ConvNeXt patient CSV contains prediction_at_development_threshold, the
script ALSO reports ConvNeXt's development-frozen Youden threshold + YOLO's
fixed zero rule. This is labelled SECONDARY and must not replace the primary
analysis merely because it looks better on the held-out cohort.

Outputs
-------
- per-patient aligned predictions and agreement flags
- publication-ready summary CSV
- JSON summary with 2,000-patient bootstrap CIs
- disagreement-direction table

No test-set threshold/model/epoch tuning occurs here.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


SEED = 42
BOOTSTRAP_SAMPLES = 2000

EXPECTED_TRAIN_IMAGES = 9541
EXPECTED_TRAIN_PATIENTS = 3354
EXPECTED_TEST_IMAGES = 2094
EXPECTED_TEST_PATIENTS = 739
EXPECTED_TEST_BENIGN_PATIENTS = 386
EXPECTED_TEST_MALIGNANT_PATIENTS = 353

EXPECTED_SELECTED_CNN = "ConvNeXtTiny"
CNN_RESULT_DIR = "ConvNeXtTiny"
YOLO_RESULT_DIR = "YOLO"

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root() -> Path:
    env = os.environ.get("THYROIDXL_PROJECT_ROOT")
    if env:
        candidate = Path(env).expanduser().resolve()
        if candidate.is_dir():
            return candidate

    for candidate in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (candidate / "results").is_dir() and (candidate / "Models").is_dir():
            return candidate

    raise RuntimeError(
        "Could not locate ThyroidXL project root. Expected Models/ and results/ "
        "above this script, or set THYROIDXL_PROJECT_ROOT."
    )


ROOT = find_project_root()
RESULTS = ROOT / "results"
OUT = RESULTS / "Combined" / "ConvNeXtTiny_YOLO_DiagnosticSelective" / "FinalOfficialTest"
OUT.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read JSON: {path}") from exc


def require_sha(value, description: str) -> str:
    sha = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise RuntimeError(f"{description} is not a valid SHA256: {value!r}")
    return sha


def load_selected_cnn_freeze() -> tuple[Path, dict]:
    path = RESULTS / "Combined" / "selected_cnn_freeze.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Missing results/Combined/selected_cnn_freeze.json. The diagnostic CNN "
            "must be prospectively frozen before official-test model selection."
        )

    payload = read_json(path)
    if payload.get("status") != "THYROIDXL_SELECTED_CNN_FROZEN_BEFORE_OFFICIAL_TEST":
        raise RuntimeError("Unexpected selected-CNN freeze status.")
    if payload.get("official_test_accessed") is not False:
        raise RuntimeError("Selected-CNN freeze does not certify untouched official test.")
    if payload.get("official_test_used_for_selection") is not False:
        raise RuntimeError("Selected-CNN freeze indicates held-out test selection.")

    selected = str(payload.get("selected_cnn", "")).strip()
    if selected != EXPECTED_SELECTED_CNN:
        raise RuntimeError(
            f"Primary dual-component CNN is expected to be {EXPECTED_SELECTED_CNN}, "
            f"but freeze selects {selected!r}. Do not change this based on test results."
        )

    require_sha(payload.get("selected_checkpoint_sha256"), "Frozen ConvNeXt checkpoint SHA")
    return path.resolve(), payload


def validate_final_summary(path: Path, model_token: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing final test summary: {path}")

    payload = read_json(path)
    if int(payload.get("official_training_images", -1)) != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError(f"{model_token}: summary does not certify 9,541 training images.")
    if int(payload.get("official_training_patients", -1)) != EXPECTED_TRAIN_PATIENTS:
        raise RuntimeError(f"{model_token}: summary does not certify 3,354 training patients.")
    if int(payload.get("official_test_images", -1)) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(f"{model_token}: summary does not certify 2,094 test images.")
    if int(payload.get("official_test_patients", -1)) != EXPECTED_TEST_PATIENTS:
        raise RuntimeError(f"{model_token}: summary does not certify 739 test patients.")
    if payload.get("official_test_used_for_tuning") is not False:
        raise RuntimeError(f"{model_token}: official_test_used_for_tuning must be false.")

    require_sha(payload.get("checkpoint_sha256"), f"{model_token} checkpoint SHA")

    status = str(payload.get("status", "")).upper().replace("-", "").replace("_", "")
    expected = model_token.upper().replace("-", "").replace("_", "")
    aliases = {
        "CONVNEXTTINY": "CONVNEXTTINY",
        "YOLOV8SSEG": "YOLO",
    }
    token = aliases.get(expected, expected)
    if token not in status or "FINAL" not in status:
        raise RuntimeError(f"{model_token}: summary status does not look final: {payload.get('status')!r}")

    return payload


def choose_column(frame: pd.DataFrame, candidates: list[str], description: str) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise RuntimeError(
        f"Could not find {description}. Tried {candidates}; available={list(frame.columns)}"
    )


def load_final_patient_outputs(result_dir: str, model_token: str) -> dict:
    directory = RESULTS / result_dir / "FinalOfficialTest"
    summary_path = directory / "test_summary.json"
    patient_path = directory / "test_patient_predictions.csv"

    if not patient_path.is_file():
        raise FileNotFoundError(f"Missing final patient predictions: {patient_path}")

    summary = validate_final_summary(summary_path, model_token)
    patient = pd.read_csv(patient_path, dtype={"patient_id": str})

    if len(patient) != EXPECTED_TEST_PATIENTS:
        raise RuntimeError(f"{model_token}: expected 739 patient rows, found {len(patient)}")
    if patient["patient_id"].duplicated().any():
        raise RuntimeError(f"{model_token}: duplicate patient IDs in final predictions.")
    if patient["label"].isin([0, 1]).all() is False:
        raise RuntimeError(f"{model_token}: labels are not binary 0/1.")

    counts = patient["label"].value_counts().to_dict()
    if counts.get(0, 0) != EXPECTED_TEST_BENIGN_PATIENTS or counts.get(1, 0) != EXPECTED_TEST_MALIGNANT_PATIENTS:
        raise RuntimeError(f"{model_token}: unexpected patient class counts: {counts}")

    return {
        "directory": directory,
        "summary_path": summary_path,
        "summary": summary,
        "patient_path": patient_path,
        "patient": patient,
    }


def binary_metrics(labels, predictions) -> dict:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity_malignant": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "specificity_benign": float(specificity),
        "precision_malignant": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "f1_malignant": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def consensus_point_estimates(frame: pd.DataFrame, cnn_pred_col: str, yolo_pred_col: str) -> dict:
    cnn_pred = frame[cnn_pred_col].to_numpy(dtype=int)
    yolo_pred = frame[yolo_pred_col].to_numpy(dtype=int)
    labels = frame["label"].to_numpy(dtype=int)

    concordant = cnn_pred == yolo_pred
    n_concordant = int(concordant.sum())
    if n_concordant == 0:
        raise RuntimeError("No concordant patients; selective metrics are undefined.")

    consensus_pred = cnn_pred[concordant]  # equal to YOLO prediction by definition
    consensus_labels = labels[concordant]

    selective = binary_metrics(consensus_labels, consensus_pred)
    cnn_full = binary_metrics(labels, cnn_pred)
    yolo_full = binary_metrics(labels, yolo_pred)

    return {
        "cases": int(len(frame)),
        "concordant_cases": n_concordant,
        "discordant_cases": int((~concordant).sum()),
        "coverage": float(concordant.mean()),
        "selective_accuracy": selective["accuracy"],
        "selective_balanced_accuracy": selective["balanced_accuracy"],
        "selective_sensitivity_malignant": selective["sensitivity_malignant"],
        "selective_specificity_benign": selective["specificity_benign"],
        "selective_precision_malignant": selective["precision_malignant"],
        "selective_f1_malignant": selective["f1_malignant"],
        "selective_mcc": selective["mcc"],
        "selective_tn": selective["tn"],
        "selective_fp": selective["fp"],
        "selective_fn": selective["fn"],
        "selective_tp": selective["tp"],
        "cnn_full_accuracy": cnn_full["accuracy"],
        "yolo_full_accuracy": yolo_full["accuracy"],
        "selective_accuracy_gain_vs_cnn_full_cohort": float(selective["accuracy"] - cnn_full["accuracy"]),
        "selective_accuracy_gain_vs_yolo_full_cohort": float(selective["accuracy"] - yolo_full["accuracy"]),
    }


def bootstrap_consensus(
    frame: pd.DataFrame,
    cnn_pred_col: str,
    yolo_pred_col: str,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(frame)

    metrics = {
        "coverage": [],
        "selective_accuracy": [],
        "selective_balanced_accuracy": [],
        "selective_sensitivity_malignant": [],
        "selective_specificity_benign": [],
        "selective_f1_malignant": [],
        "selective_mcc": [],
    }

    for _ in range(int(samples)):
        idx = rng.integers(0, n, size=n)
        boot = frame.iloc[idx]
        cnn = boot[cnn_pred_col].to_numpy(dtype=int)
        yolo = boot[yolo_pred_col].to_numpy(dtype=int)
        labels = boot["label"].to_numpy(dtype=int)
        concordant = cnn == yolo

        metrics["coverage"].append(float(concordant.mean()))
        if not concordant.any():
            continue

        retained_labels = labels[concordant]
        retained_pred = cnn[concordant]
        if len(np.unique(retained_labels)) < 2:
            # Accuracy is still defined, but class-balanced metrics are not stable.
            metrics["selective_accuracy"].append(float(accuracy_score(retained_labels, retained_pred)))
            continue

        bundle = binary_metrics(retained_labels, retained_pred)
        for key in (
            "selective_accuracy",
            "selective_balanced_accuracy",
            "selective_sensitivity_malignant",
            "selective_specificity_benign",
            "selective_f1_malignant",
            "selective_mcc",
        ):
            source = key.replace("selective_", "")
            metrics[key].append(float(bundle[source]))

    result = {}
    for key, values in metrics.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        result[key] = {
            "ci95_low": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
            "ci95_high": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
            "bootstrap_samples_valid": int(len(arr)),
        }
    return result


def subgroup_summary(frame: pd.DataFrame, cnn_pred_col: str, yolo_pred_col: str, seed_offset: int) -> dict:
    result = {}
    for label, name in ((0, "benign"), (1, "malignant")):
        group = frame[frame["label"] == label].copy()
        cnn = group[cnn_pred_col].to_numpy(dtype=int)
        yolo = group[yolo_pred_col].to_numpy(dtype=int)
        labels = group["label"].to_numpy(dtype=int)
        concordant_point = cnn == yolo
        if not concordant_point.any():
            raise RuntimeError(f"No concordant patients in {name} subgroup.")

        selective_accuracy = float(accuracy_score(labels[concordant_point], cnn[concordant_point]))
        cnn_accuracy = float(accuracy_score(labels, cnn))
        yolo_accuracy = float(accuracy_score(labels, yolo))

        # For a single-class subgroup, balanced accuracy/MCC are not useful.
        # Keep coverage and selective accuracy, plus class-specific correctness.
        rng = np.random.default_rng(SEED + seed_offset + label)
        cov, acc = [], []
        n = len(group)
        for _ in range(BOOTSTRAP_SAMPLES):
            idx = rng.integers(0, n, size=n)
            boot = group.iloc[idx]
            concordant = (
                boot[cnn_pred_col].to_numpy(dtype=int)
                == boot[yolo_pred_col].to_numpy(dtype=int)
            )
            cov.append(float(concordant.mean()))
            if concordant.any():
                pred = boot.loc[concordant, cnn_pred_col].to_numpy(dtype=int)
                lab = boot.loc[concordant, "label"].to_numpy(dtype=int)
                acc.append(float(accuracy_score(lab, pred)))

        result[name] = {
            "point_estimates": {
                "cases": int(len(group)),
                "concordant_cases": int(concordant_point.sum()),
                "discordant_cases": int((~concordant_point).sum()),
                "coverage": float(concordant_point.mean()),
                "selective_accuracy": selective_accuracy,
                "cnn_full_accuracy": cnn_accuracy,
                "yolo_full_accuracy": yolo_accuracy,
                "selective_accuracy_gain_vs_cnn_full_cohort": float(selective_accuracy - cnn_accuracy),
                "selective_accuracy_gain_vs_yolo_full_cohort": float(selective_accuracy - yolo_accuracy),
            },
            "bootstrap_ci95": {
                "coverage": {
                    "ci95_low": float(np.quantile(cov, 0.025)),
                    "ci95_high": float(np.quantile(cov, 0.975)),
                },
                "selective_accuracy": {
                    "ci95_low": float(np.quantile(acc, 0.025)),
                    "ci95_high": float(np.quantile(acc, 0.975)),
                },
            },
        }
    return result


def disagreement_table(frame: pd.DataFrame, cnn_pred_col: str, yolo_pred_col: str) -> pd.DataFrame:
    cnn = frame[cnn_pred_col].astype(int)
    yolo = frame[yolo_pred_col].astype(int)

    group = np.select(
        [
            (cnn == 0) & (yolo == 0),
            (cnn == 1) & (yolo == 1),
            (cnn == 1) & (yolo == 0),
            (cnn == 0) & (yolo == 1),
        ],
        [
            "concordant_benign",
            "concordant_malignant",
            "convnext_malignant_yolo_benign",
            "convnext_benign_yolo_malignant",
        ],
        default="unexpected",
    )

    temp = frame[["patient_id", "label"]].copy()
    temp["agreement_group"] = group

    rows = []
    for name, part in temp.groupby("agreement_group", sort=False):
        rows.append({
            "agreement_group": name,
            "n": int(len(part)),
            "reference_benign": int((part["label"] == 0).sum()),
            "reference_malignant": int((part["label"] == 1).sum()),
        })
    return pd.DataFrame(rows)


def analyse_rule(
    aligned: pd.DataFrame,
    *,
    rule_name: str,
    cnn_pred_col: str,
    yolo_pred_col: str,
    seed_offset: int,
) -> tuple[dict, pd.DataFrame]:
    point = consensus_point_estimates(aligned, cnn_pred_col, yolo_pred_col)
    boot = bootstrap_consensus(
        aligned,
        cnn_pred_col,
        yolo_pred_col,
        seed=SEED + seed_offset,
    )
    subgroups = subgroup_summary(aligned, cnn_pred_col, yolo_pred_col, seed_offset + 100)
    disagreements = disagreement_table(aligned, cnn_pred_col, yolo_pred_col)

    return {
        "rule_name": rule_name,
        "cnn_prediction_column": cnn_pred_col,
        "yolo_prediction_column": yolo_pred_col,
        "overall": {
            "point_estimates": point,
            "bootstrap_ci95": boot,
        },
        "by_diagnosis": subgroups,
        "disagreement_groups": disagreements.to_dict(orient="records"),
    }, disagreements


def add_rule_flags(frame: pd.DataFrame, cnn_pred_col: str, yolo_pred_col: str, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    out[f"{prefix}__cnn_prediction"] = out[cnn_pred_col].astype(int)
    out[f"{prefix}__yolo_prediction"] = out[yolo_pred_col].astype(int)
    out[f"{prefix}__concordant"] = (
        out[f"{prefix}__cnn_prediction"] == out[f"{prefix}__yolo_prediction"]
    ).astype(int)
    out[f"{prefix}__consensus_prediction"] = np.where(
        out[f"{prefix}__concordant"] == 1,
        out[f"{prefix}__cnn_prediction"],
        np.nan,
    )
    out[f"{prefix}__consensus_correct"] = np.where(
        out[f"{prefix}__concordant"] == 1,
        (out[f"{prefix}__cnn_prediction"] == out["label"]).astype(int),
        np.nan,
    )
    return out


def main():
    print("=" * 88)
    print("THYROIDXL — PRIMARY CONVNEXT-TINY ↔ YOLO DIAGNOSTIC / SELECTIVE ANALYSIS")
    print("=" * 88)
    print("Project root:", ROOT)
    print("No held-out model or threshold selection is performed here.")
    print()

    freeze_path, freeze = load_selected_cnn_freeze()
    cnn = load_final_patient_outputs(CNN_RESULT_DIR, "ConvNeXtTiny")
    yolo = load_final_patient_outputs(YOLO_RESULT_DIR, "YOLOv8sSeg")

    frozen_sha = require_sha(
        freeze.get("selected_checkpoint_sha256"),
        "Frozen selected-CNN checkpoint SHA",
    )
    evaluated_cnn_sha = require_sha(
        cnn["summary"].get("checkpoint_sha256"),
        "Evaluated ConvNeXt checkpoint SHA",
    )
    if frozen_sha != evaluated_cnn_sha:
        raise RuntimeError(
            "The prospectively frozen ConvNeXt checkpoint is not the checkpoint "
            "used for final held-out evaluation."
        )

    cnn_df = cnn["patient"].copy()
    yolo_df = yolo["patient"].copy()

    cnn_primary = choose_column(
        cnn_df,
        ["prediction_primary", "prediction_at_0_5"],
        "ConvNeXt primary prediction column",
    )
    yolo_primary = choose_column(
        yolo_df,
        ["prediction_primary", "prediction_at_zero"],
        "YOLO primary prediction column",
    )

    cnn_keep = ["patient_id", "label", cnn_primary]
    if "probability_malignant" in cnn_df.columns:
        cnn_keep.append("probability_malignant")
    if "prediction_at_development_threshold" in cnn_df.columns:
        cnn_keep.append("prediction_at_development_threshold")

    yolo_keep = ["patient_id", "label", yolo_primary]
    if "malignant_score" in yolo_df.columns:
        yolo_keep.append("malignant_score")

    cnn_sub = cnn_df[cnn_keep].copy().rename(columns={
        cnn_primary: "convnext_prediction_primary",
        "probability_malignant": "convnext_probability_malignant",
        "prediction_at_development_threshold": "convnext_prediction_development_threshold",
    })
    yolo_sub = yolo_df[yolo_keep].copy().rename(columns={
        yolo_primary: "yolo_prediction_primary",
        "malignant_score": "yolo_malignant_score",
    })

    aligned = cnn_sub.merge(
        yolo_sub,
        on=["patient_id", "label"],
        how="inner",
        validate="one_to_one",
    )
    if len(aligned) != EXPECTED_TEST_PATIENTS:
        raise RuntimeError(
            f"ConvNeXt/YOLO patient alignment produced {len(aligned)} rows; expected 739."
        )

    primary, primary_disagreements = analyse_rule(
        aligned,
        rule_name="PRIMARY: ConvNeXt fixed 0.5 + YOLO signed score > 0",
        cnn_pred_col="convnext_prediction_primary",
        yolo_pred_col="yolo_prediction_primary",
        seed_offset=0,
    )
    aligned = add_rule_flags(
        aligned,
        "convnext_prediction_primary",
        "yolo_prediction_primary",
        "primary",
    )

    secondary = None
    secondary_disagreements = None
    if "convnext_prediction_development_threshold" in aligned.columns:
        secondary, secondary_disagreements = analyse_rule(
            aligned,
            rule_name=(
                "SECONDARY PRE-SPECIFIED: ConvNeXt development-frozen threshold + "
                "YOLO signed score > 0"
            ),
            cnn_pred_col="convnext_prediction_development_threshold",
            yolo_pred_col="yolo_prediction_primary",
            seed_offset=1000,
        )
        aligned = add_rule_flags(
            aligned,
            "convnext_prediction_development_threshold",
            "yolo_prediction_primary",
            "secondary_development_threshold",
        )

    summary = {
        "status": "THYROIDXL_CONVNEXTTINY_YOLO_PRIMARY_DIAGNOSTIC_SELECTIVE_ANALYSIS",
        "official_test_used_for_tuning": False,
        "official_test_patients": EXPECTED_TEST_PATIENTS,
        "selected_cnn_freeze": str(freeze_path),
        "selected_cnn": freeze["selected_cnn"],
        "selected_cnn_freeze_checkpoint_sha256": frozen_sha,
        "convnext_final_summary": str(cnn["summary_path"]),
        "convnext_final_checkpoint_sha256": evaluated_cnn_sha,
        "yolo_final_summary": str(yolo["summary_path"]),
        "yolo_final_checkpoint_sha256": require_sha(
            yolo["summary"].get("checkpoint_sha256"),
            "YOLO final checkpoint SHA",
        ),
        "primary_analysis": primary,
        "secondary_development_threshold_analysis": secondary,
        "interpretation_policy": {
            "primary_pair": "ConvNeXt-Tiny + YOLOv8s-seg, prospectively selected from development only",
            "primary_decision_rules": "ConvNeXt fixed 0.5; YOLO signed evidence > 0",
            "secondary_rule": (
                "ConvNeXt development-frozen threshold may be reported as a pre-specified "
                "secondary sensitivity analysis; do not promote it to primary based on held-out results."
            ),
            "selective_accuracy_meaning": (
                "Accuracy among patients retained because the two diagnostic predictions agree; "
                "coverage is the fraction of all held-out patients retained."
            ),
        },
    }

    aligned_path = OUT / "convnext_yolo_agreement_patients.csv"
    summary_path = OUT / "convnext_yolo_diagnostic_selective_summary.json"
    table_path = OUT / "convnext_yolo_diagnostic_selective_publication_table.csv"
    disagreement_path = OUT / "convnext_yolo_primary_disagreement_groups.csv"

    aligned.to_csv(aligned_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    primary_disagreements.to_csv(disagreement_path, index=False)
    if secondary_disagreements is not None:
        secondary_disagreements.to_csv(
            OUT / "convnext_yolo_secondary_disagreement_groups.csv",
            index=False,
        )

    rows = []
    for analysis_label, payload in (
        ("primary", primary),
        ("secondary_development_threshold", secondary),
    ):
        if payload is None:
            continue
        overall = payload["overall"]["point_estimates"]
        ci = payload["overall"]["bootstrap_ci95"]
        rows.append({
            "analysis": analysis_label,
            "rule": payload["rule_name"],
            "n_patients": overall["cases"],
            "n_concordant": overall["concordant_cases"],
            "coverage": overall["coverage"],
            "coverage_ci95_low": ci["coverage"]["ci95_low"],
            "coverage_ci95_high": ci["coverage"]["ci95_high"],
            "selective_accuracy": overall["selective_accuracy"],
            "selective_accuracy_ci95_low": ci["selective_accuracy"]["ci95_low"],
            "selective_accuracy_ci95_high": ci["selective_accuracy"]["ci95_high"],
            "selective_sensitivity_malignant": overall["selective_sensitivity_malignant"],
            "selective_specificity_benign": overall["selective_specificity_benign"],
            "selective_f1_malignant": overall["selective_f1_malignant"],
            "selective_mcc": overall["selective_mcc"],
            "cnn_full_accuracy": overall["cnn_full_accuracy"],
            "yolo_full_accuracy": overall["yolo_full_accuracy"],
            "gain_vs_cnn_full_cohort": overall["selective_accuracy_gain_vs_cnn_full_cohort"],
            "gain_vs_yolo_full_cohort": overall["selective_accuracy_gain_vs_yolo_full_cohort"],
        })
    pd.DataFrame(rows).to_csv(table_path, index=False)

    overall = primary["overall"]["point_estimates"]
    ci = primary["overall"]["bootstrap_ci95"]
    print("PRIMARY diagnostic consensus")
    print(f"  Coverage:           {overall['coverage'] * 100:.2f}% "
          f"[{ci['coverage']['ci95_low'] * 100:.2f}, {ci['coverage']['ci95_high'] * 100:.2f}]")
    print(f"  Selective accuracy: {overall['selective_accuracy'] * 100:.2f}% "
          f"[{ci['selective_accuracy']['ci95_low'] * 100:.2f}, {ci['selective_accuracy']['ci95_high'] * 100:.2f}]")
    print(f"  Selective sens/spec: {overall['selective_sensitivity_malignant'] * 100:.2f}% / "
          f"{overall['selective_specificity_benign'] * 100:.2f}%")
    print(f"  Selective F1/MCC:    {overall['selective_f1_malignant']:.4f} / {overall['selective_mcc']:.4f}")
    print()

    if secondary is not None:
        sec = secondary["overall"]["point_estimates"]
        sci = secondary["overall"]["bootstrap_ci95"]
        print("SECONDARY pre-specified development-threshold consensus")
        print(f"  Coverage:           {sec['coverage'] * 100:.2f}% "
              f"[{sci['coverage']['ci95_low'] * 100:.2f}, {sci['coverage']['ci95_high'] * 100:.2f}]")
        print(f"  Selective accuracy: {sec['selective_accuracy'] * 100:.2f}% "
              f"[{sci['selective_accuracy']['ci95_low'] * 100:.2f}, {sci['selective_accuracy']['ci95_high'] * 100:.2f}]")
        print()

    print("Saved:")
    print("  Patients:", aligned_path)
    print("  Summary:", summary_path)
    print("  Publication table:", table_path)
    print("  Primary disagreement groups:", disagreement_path)


if __name__ == "__main__":
    main()
