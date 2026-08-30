from __future__ import annotations

import hashlib
import tkinter
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from ultralytics import YOLO

import logic

def _find_project_root(script_file: str | Path) -> Path:
    script_dir = Path(script_file).resolve().parent

    import os
    env = os.environ.get("THYROIDXL_PROJECT_ROOT")
    if env:
        candidate = Path(env).expanduser().resolve()
        if (
            (candidate / "Models").is_dir()
            or (candidate / "models").is_dir()
        ):
            return candidate
        raise RuntimeError(
            "THYROIDXL_PROJECT_ROOT is set, but no Models/ directory was found:\n"
            f"  {candidate}"
        )

    for candidate in [script_dir, *script_dir.parents]:
        if (
            (candidate / "Models").is_dir()
            or (candidate / "models").is_dir()
        ):
            return candidate

    raise RuntimeError(
        "Could not locate the ThyroidXL project root.\n"
        "Expected to find a Models/ directory in this script's folder or one "
        "of its parent folders.\n\n"
        "You can also set the THYROIDXL_PROJECT_ROOT environment variable."
    )

def _models_root(project_root: Path) -> Path:
    for name in ("Models", "models"):
        candidate = Path(project_root) / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No Models/ directory found under:\n  {project_root}"
    )

PROJECT_ROOT = _find_project_root(__file__)
MODELS_ROOT = _models_root(PROJECT_ROOT)

PIPELINE_BUILD = "2026-08-26-BTXRD-2x2-positive-display-scores-v6.4"

CLASSIFIER_CHECKPOINT: Path | None = None
YOLO_CHECKPOINT: Path | None = None

DISPLAY_SIZE = 640
YOLO_IMAGE_SIZE = 640

YOLO_RETRIEVAL_CONFIDENCE = 0.001
YOLO_NMS_IOU = 0.70
YOLO_MAX_DETECTIONS = 300

CNN_FRAME_THRESHOLD = 0.50

CAM_METHODS = {
    "Grad-CAM": GradCAM,
    "Grad-CAM++": GradCAMPlusPlus,
    "Layer-CAM": LayerCAM,
}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

FINAL_MOBILENET_STATUS = (
    "THYROIDXL_MOBILENET_FINAL_OFFICIAL_TRAIN_"
    "REFIT_ONEFOLD_SELECTED"
)

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

def _find_mobilenet_checkpoint() -> Path:
    mobile_dir_candidates = [
        MODELS_ROOT / "MobileNetV3",
        MODELS_ROOT / "MobilenetV3",
        MODELS_ROOT / "mobilenetv3",
        MODELS_ROOT,
    ]

    search_roots = [
        p for p in mobile_dir_candidates
        if p.is_dir()
    ]

    candidates = []
    seen = set()

    for root in search_roots:
        for path in root.rglob("*.pt"):
            resolved = path.resolve()
            key = str(resolved).lower()
            if key in seen:
                continue
            seen.add(key)

            name = resolved.name.lower()
            if "mobile" not in name and root == MODELS_ROOT:
                continue

            try:
                checkpoint = _torch_load_checkpoint(resolved)
            except Exception:
                continue

            if not isinstance(checkpoint, dict):
                continue
            if "state_dict" not in checkpoint:
                continue
            if str(checkpoint.get("dataset", "")).strip().lower() != "thyroidxl":
                continue

            model_name = str(checkpoint.get("model_name", "")).lower()
            variant = str(checkpoint.get("model_variant", "")).lower()

            if (
                "mobilenet" not in model_name
                and "modelb_v2" not in variant
                and "mobilenet" not in variant
            ):
                continue

            candidates.append((resolved, checkpoint))

    if not candidates:
        raise FileNotFoundError(
            "Could not find a valid ThyroidXL MobileNetV3 checkpoint.\n\n"
            f"Searched under:\n  {MODELS_ROOT}\n\n"
            "Expected a checkpoint containing dataset='ThyroidXL', a state_dict, "
            "and MobileNetV3/ModelB_V2 metadata."
        )

    exact_final = [
        (path, ckpt)
        for path, ckpt in candidates
        if ckpt.get("status") == FINAL_MOBILENET_STATUS
    ]

    if len(exact_final) == 1:
        return exact_final[0][0]

    if len(exact_final) > 1:
        raise RuntimeError(
            "Multiple final MobileNet checkpoints were found:\n  "
            + "\n  ".join(str(path) for path, _ in exact_final)
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            "officialtrain9541" not in item[0].name.lower(),
            "final" not in item[0].name.lower(),
            "onefoldselected" not in item[0].name.lower(),
            len(str(item[0])),
            str(item[0]).lower(),
        ),
    )

    return ranked[0][0]

def _find_yolo_checkpoint() -> Path:
    yolo_dir_candidates = [
        MODELS_ROOT / "YOLOv8sSeg",
        MODELS_ROOT / "YOLOv8s-seg",
        MODELS_ROOT / "yolov8sseg",
        MODELS_ROOT,
    ]

    candidates = []
    seen = set()

    for root in yolo_dir_candidates:
        if not root.is_dir():
            continue

        for path in root.rglob("*.pt"):
            resolved = path.resolve()
            key = str(resolved).lower()
            if key in seen:
                continue
            seen.add(key)

            name = resolved.name.lower()
            if root == MODELS_ROOT and "yolo" not in name:
                continue

            candidates.append(resolved)

    if not candidates:
        raise FileNotFoundError(
            "Could not find a YOLOv8s-seg checkpoint.\n\n"
            f"Searched under:\n  {MODELS_ROOT}"
        )

    ranked = sorted(
        candidates,
        key=lambda p: (
            "selected" not in p.name.lower(),
            "maskmap" not in p.name.lower()
            and "mask" not in p.name.lower(),
            "final" not in p.name.lower(),
            len(str(p)),
            str(p).lower(),
        ),
    )

    return ranked[0]

def _load_mobilenet_classifier(checkpoint_path: Path):
    checkpoint = _torch_load_checkpoint(checkpoint_path)

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected dictionary checkpoint, got {type(checkpoint)!r}"
        )

    required = {
        "dataset",
        "model_name",
        "image_size",
        "state_dict",
    }
    missing = sorted(required - set(checkpoint))

    if missing:
        raise KeyError(
            f"MobileNet checkpoint is missing required keys: {missing}"
        )

    if str(checkpoint["dataset"]).strip().lower() != "thyroidxl":
        raise RuntimeError(
            f"Expected ThyroidXL checkpoint, got {checkpoint['dataset']!r}."
        )

    model = logic.MobileNetV3MultiscaleDiceBCE(
        model_name=str(checkpoint["model_name"]),
        image_size=int(checkpoint["image_size"]),
        drop_rate=float(
            checkpoint.get(
                "drop_rate",
                logic.DEFAULT_DROP_RATE,
            )
        ),
    )

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    model = model.to(logic.DEVICE).eval()

    return model, checkpoint

class Visualiser:

    def __init__(self):
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.edgecolor": "none",
                "axes.titlesize": 15,
                "font.family": "sans-serif",
            }
        )

    @staticmethod
    def _hide_axis(axis):
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    @staticmethod
    def _write(
        axis,
        x,
        y,
        text,
        **kwargs,
    ):
        axis.text(
            x,
            y,
            text,
            transform=axis.transAxes,
            ha=kwargs.pop("ha", "left"),
            va=kwargs.pop("va", "top"),
            **kwargs,
        )

    @staticmethod
    def _agreement_value(value):
        if value is None:
            return "N/A", "#777777"
        if bool(value):
            return "TRUE", "#1B7F3A"
        return "FALSE", "#D62728"

    def plot(
        self,
        plot_data,
        report,
    ):
        display_rgb = plot_data["display_rgb"]
        yolo_overlay = plot_data["yolo_overlay"]
        layercam_overlay = plot_data["cam_overlays"]["Layer-CAM"]

        figure = plt.figure(
            figsize=(13.5, 10.4),
            facecolor="white",
        )

        grid = GridSpec(
            2,
            2,
            figure=figure,
            width_ratios=[1.0, 1.0],
            height_ratios=[1.0, 1.0],
            wspace=0.09,
            hspace=0.14,
        )

        ax_original = figure.add_subplot(
            grid[0, 0]
        )
        ax_summary = figure.add_subplot(
            grid[0, 1]
        )
        ax_yolo = figure.add_subplot(
            grid[1, 0]
        )
        ax_layercam = figure.add_subplot(
            grid[1, 1]
        )

        ax_original.imshow(
            display_rgb
        )
        ax_original.set_title(
            "Thyroid Ultrasound",
            fontsize=15,
            weight="bold",
            pad=8,
        )
        self._hide_axis(
            ax_original
        )

        self._summary_panel(
            ax_summary,
            report,
        )

        ax_yolo.imshow(
            yolo_overlay
        )
        ax_yolo.set_title(
            "YOLOv8s-seg Localisation",
            fontsize=15,
            weight="bold",
            pad=8,
        )
        self._hide_axis(
            ax_yolo
        )

        if report[
            "yolo_detection_available"
        ]:
            yolo_caption = (
                f"{report['yolo_label']} "
                f"(Detection score: "
                f"{report['yolo_top_detection_confidence']:.3f})"
            )
        else:
            yolo_caption = (
                f"{report['yolo_label']} "
                "(No localisation)"
            )

        ax_yolo.text(
            0.50,
            -0.065,
            yolo_caption,
            transform=ax_yolo.transAxes,
            ha="center",
            va="top",
            fontsize=11.5,
            weight="bold",
            color="#222222",
            clip_on=False,
        )

        ax_layercam.imshow(
            layercam_overlay
        )
        ax_layercam.set_title(
            "Layer-CAM",
            fontsize=15,
            weight="bold",
            pad=8,
        )
        self._hide_axis(
            ax_layercam
        )

        malignancy_probability = float(
            report[
                "cnn_malignancy_score"
            ]
        )

        cnn_display_score = (
            malignancy_probability
            if str(
                report[
                    "cnn_label"
                ]
            ).strip().lower()
            == "malignant"
            else 1.0
            - malignancy_probability
        )

        cnn_caption = (
            f"{report['cnn_label']} "
            f"(Classification score: "
            f"{cnn_display_score:.3f})"
        )

        ax_layercam.text(
            0.50,
            -0.065,
            cnn_caption,
            transform=ax_layercam.transAxes,
            ha="center",
            va="top",
            fontsize=11.5,
            weight="bold",
            color="#222222",
            clip_on=False,
        )

        figure.subplots_adjust(
            top=0.955,
            bottom=0.095,
            left=0.040,
            right=0.965,
            wspace=0.080,
            hspace=0.135,
        )

        plt.show()

    def _summary_panel(
        self,
        axis,
        report,
    ):
        axis.set_xlim(
            0,
            1,
        )
        axis.set_ylim(
            0,
            1,
        )
        axis.axis(
            "off"
        )

        axis.set_title(
            "Model Output Summary",
            fontsize=15,
            weight="bold",
            pad=8,
        )

        axis.axhline(
            0.965,
            xmin=0.02,
            xmax=0.98,
            color="#D9D9D9",
            linewidth=1.6,
        )

        y = 0.900

        self._write(
            axis,
            0.02,
            y,
            "Component 1: CNN Classification",
            fontsize=12.0,
            weight="bold",
        )

        y -= 0.085

        cnn_malignancy_probability = float(
            report[
                "cnn_malignancy_score"
            ]
        )

        cnn_summary_score = (
            cnn_malignancy_probability
            if str(
                report[
                    "cnn_label"
                ]
            ).strip().lower()
            == "malignant"
            else 1.0
            - cnn_malignancy_probability
        )

        self._write(
            axis,
            0.045,
            y,
            (
                f"Output: {report['cnn_label']} "
                f"(Classification score: "
                f"{cnn_summary_score:.3f})"
            ),
            fontsize=11.0,
        )

        y -= 0.085

        axis.axhline(
            y + 0.025,
            xmin=0.02,
            xmax=0.98,
            color="#E8E8E8",
            linewidth=1.0,
            linestyle="--",
        )

        self._write(
            axis,
            0.02,
            y,
            "Component 2: YOLOv8s-seg",
            fontsize=12.0,
            weight="bold",
        )

        y -= 0.085

        if report[
            "yolo_detection_available"
        ]:
            yolo_text = (
                f"Output: {report['yolo_label']} "
                f"(Detection score: "
                f"{report['yolo_top_detection_confidence']:.3f})"
            )

        else:
            yolo_text = (
                f"Output: {report['yolo_label']} "
                "(No localisation)"
            )

        self._write(
            axis,
            0.045,
            y,
            yolo_text,
            fontsize=11.0,
        )

        y -= 0.085

        axis.axhline(
            y + 0.025,
            xmin=0.02,
            xmax=0.98,
            color="#E8E8E8",
            linewidth=1.0,
            linestyle="--",
        )

        self._write(
            axis,
            0.02,
            y,
            "Component Agreement",
            fontsize=12.0,
            weight="bold",
        )

        y -= 0.085

        agreement_text, agreement_colour = (
            self._agreement_value(
                report[
                    "diagnostic_agreement"
                ]
            )
        )

        self._write(
            axis,
            0.045,
            y,
            "Diagnostic Agreement:",
            fontsize=11.0,
        )

        self._write(
            axis,
            0.965,
            y,
            agreement_text,
            fontsize=11.0,
            weight="bold",
            color=agreement_colour,
            ha="right",
        )

        y -= 0.080

        location_text, location_colour = (
            self._agreement_value(
                report[
                    "location_agreement"
                ]
            )
        )

        self._write(
            axis,
            0.045,
            y,
            "Location Agreement:",
            fontsize=11.0,
        )

        self._write(
            axis,
            0.965,
            y,
            location_text,
            fontsize=11.0,
            weight="bold",
            color=location_colour,
            ha="right",
        )

        summary = report[
            "evidence_summary"
        ]

        if report[
            "yolo_detection_available"
        ] is False:
            box_text = (
                "Incomplete Spatial Evidence"
            )
        elif (
            report[
                "diagnostic_agreement"
            ]
            and report[
                "location_agreement"
            ]
        ):
            box_text = (
                "Concordant Model Outputs"
            )
        elif report[
            "diagnostic_agreement"
        ]:
            box_text = (
                "Diagnostic Agreement, Location Mismatch"
            )
        else:
            box_text = (
                "Discordant Model Outputs"
            )

        box_y = 0.080
        box_height = 0.145

        axis.add_patch(
            plt.Rectangle(
                (
                    0.02,
                    box_y,
                ),
                0.96,
                box_height,
                transform=axis.transAxes,
                facecolor=summary[
                    "color"
                ],
                edgecolor="none",
                alpha=0.11,
            )
        )

        axis.text(
            0.50,
            box_y
            + box_height
            / 2.0,
            box_text,
            transform=axis.transAxes,
            fontsize=13.2,
            weight="bold",
            ha="center",
            va="center",
            color=summary[
                "color"
            ],
        )

class App:
    def __init__(
        self,
        root,
        classifier,
        classifier_checkpoint,
        classifier_wrapper,
        yolo,
        xai_configs,
        xai_freeze_path,
        yolo_config,
        yolo_freeze_path,
    ):
        self.root = root
        self.classifier = classifier
        self.classifier_checkpoint = classifier_checkpoint
        self.classifier_wrapper = classifier_wrapper
        self.yolo = yolo
        self.xai_configs = xai_configs
        self.xai_freeze_path = xai_freeze_path
        self.yolo_config = yolo_config
        self.yolo_freeze_path = yolo_freeze_path
        self.visualiser = Visualiser()

        self.root.title(
            "ThyroidXL Dual-Component Evidence Visualisation"
        )
        self.root.geometry("820x220")
        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding="14")
        main.pack(fill=tkinter.BOTH, expand=True)

        ttk.Label(
            main,
            text="ThyroidXL Dual-Component Evidence Visualisation",
            font=("TkDefaultFont", 13, "bold"),
        ).pack(anchor="w", pady=(0, 12))

        selection = ttk.LabelFrame(
            main,
            text="Ultrasound image",
            padding="10",
        )
        selection.pack(fill=tkinter.X)

        self.image_path_var = tkinter.StringVar()

        ttk.Entry(
            selection,
            textvariable=self.image_path_var,
        ).pack(
            side=tkinter.LEFT,
            fill=tkinter.X,
            expand=True,
            padx=(0, 8),
        )

        ttk.Button(
            selection,
            text="Browse…",
            command=self._choose_image,
        ).pack(side=tkinter.LEFT)

        ttk.Button(
            main,
            text="Run evidence visualisation",
            command=self._run_visualisation,
        ).pack(
            fill=tkinter.X,
            pady=(14, 0),
            ipady=4,
        )

        ttk.Label(
            main,
            foreground="#555555",
        ).pack(anchor="w", pady=(10, 0))

    def _choose_image(self):
        path = filedialog.askopenfilename(
            title="Choose a thyroid ultrasound image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.image_path_var.set(path)

    def _run_visualisation(self):
        try:
            image_path = Path(
                self.image_path_var.get().strip()
            )

            if not image_path.is_file():
                raise FileNotFoundError(
                    "Choose a valid thyroid ultrasound image first."
                )

            display_rgb, image_tensor = (
                logic.prepare_display_and_tensor(
                    image_path=image_path,
                    input_size=int(
                        self.classifier_checkpoint["image_size"]
                    ),
                    display_size=DISPLAY_SIZE,
                )
            )

            display_shape = display_rgb.shape[:2]

            classifier_result = self._run_classifier(
                image_tensor,
                display_shape,
            )
            yolo_result = self._run_yolo(
                image_path,
                display_shape,
            )
            cam_maps = self._generate_cams(
                image_tensor,
                classifier_result["prediction"],
                display_shape,
            )

            plot_data = self._prepare_plot_data(
                display_rgb,
                classifier_result,
                yolo_result,
                cam_maps,
            )

            report = self._create_report(
                image_path.name,
                classifier_result,
                yolo_result,
                cam_maps,
            )

            self.visualiser.plot(plot_data, report)

        except Exception as error:
            traceback.print_exc()
            messagebox.showerror(
                "Visualisation error",
                str(error),
            )

    def _run_classifier(
        self,
        image_tensor,
        display_shape,
    ):
        with torch.inference_mode():
            logits, segmentation_logits = self.classifier(
                image_tensor
            )

            malignancy_score = float(
                torch.sigmoid(logits.reshape(-1)[0]).item()
            )

            segmentation_probability = torch.sigmoid(
                segmentation_logits
            )[0, 0]

        prediction = int(
            malignancy_score >= CNN_FRAME_THRESHOLD
        )
        label = "Malignant" if prediction == 1 else "Benign"

        mask_512 = (
            segmentation_probability
            .detach()
            .cpu()
            .numpy()
            >= 0.5
        ).astype(np.uint8)

        display_h = int(display_shape[0])
        display_w = int(display_shape[1])

        display_mask = (
            logic.unpad_square_to_shape(
                mask_512,
                target_height=display_h,
                target_width=display_w,
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        ).astype(np.uint8)

        return {
            "prediction": prediction,
            "label": label,
            "malignancy_score": malignancy_score,
            "threshold": CNN_FRAME_THRESHOLD,
            "mask": display_mask,
            "mask_fraction": float(mask_512.mean()),
        }

    @staticmethod
    def _yolo_class_ids(result):
        names = result.names
        if not isinstance(names, dict):
            names = {
                i: str(value)
                for i, value in enumerate(names)
            }

        normalized = {
            int(k): str(v).strip().lower()
            for k, v in names.items()
        }

        benign = [
            class_id
            for class_id, name in normalized.items()
            if "benign" in name
        ]
        malignant = [
            class_id
            for class_id, name in normalized.items()
            if "malignant" in name
        ]

        if len(benign) != 1 or len(malignant) != 1:
            raise RuntimeError(
                "YOLO checkpoint must contain exactly two ThyroidXL classes: "
                f"benign and malignant. Found: {names}"
            )

        return benign[0], malignant[0]

    def _run_yolo(
        self,
        image_path: Path,
        display_shape,
    ):
        original_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )
        if original_bgr is None:
            raise ValueError(
                f"OpenCV could not read image: {image_path}"
            )

        height, width = original_bgr.shape[:2]

        result = self.yolo.predict(
            source=str(image_path),
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_RETRIEVAL_CONFIDENCE,
            iou=YOLO_NMS_IOU,
            max_det=YOLO_MAX_DETECTIONS,
            device=(
                0
                if torch.cuda.is_available()
                else "cpu"
            ),
            augment=bool(self.yolo_config["augment"]),
            retina_masks=True,
            verbose=False,
        )[0]

        benign_id, malignant_id = self._yolo_class_ids(result)

        boxes = result.boxes

        display_h = int(display_shape[0])
        display_w = int(display_shape[1])

        display_mask = np.zeros(
            (display_h, display_w),
            dtype=np.uint8,
        )

        if boxes is None or len(boxes) == 0:
            return {
                "detection_available": False,
                "label": None,
                "prediction": None,
                "signed_score": 0.0,
                "threshold": float(
                    self.yolo_config["image_threshold"]
                ),
                "benign_score": 0.0,
                "malignant_score": 0.0,
                "top_detection_class": None,
                "top_detection_confidence": 0.0,
                "mask": display_mask,
            }

        classes = (
            boxes.cls.detach().cpu().numpy().astype(int)
        )
        confidences = (
            boxes.conf.detach().cpu().numpy().astype(float)
        )

        benign_values = confidences[classes == benign_id]
        malignant_values = confidences[classes == malignant_id]

        p_b = (
            float(benign_values.max())
            if benign_values.size
            else 0.0
        )
        p_m = (
            float(malignant_values.max())
            if malignant_values.size
            else 0.0
        )

        signed_score = float(p_m - p_b)
        threshold = float(
            self.yolo_config["image_threshold"]
        )

        prediction = int(signed_score >= threshold)
        label = "Malignant" if prediction == 1 else "Benign"

        top_index = int(np.argmax(confidences))
        top_class_id = int(classes[top_index])
        top_confidence = float(confidences[top_index])
        top_class_name = str(
            result.names[top_class_id]
        ).strip().title()

        original_mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        polygon_added = False
        if (
            result.masks is not None
            and top_index < len(result.masks.xy)
        ):
            polygon = np.asarray(
                result.masks.xy[top_index],
                dtype=np.float32,
            )

            if (
                polygon.ndim == 2
                and polygon.shape[0] >= 3
            ):
                polygon = np.round(polygon).astype(np.int32)
                cv2.fillPoly(
                    original_mask,
                    [polygon],
                    1,
                )
                polygon_added = True

        if not polygon_added:
            x1, y1, x2, y2 = (
                boxes.xyxy[top_index]
                .detach()
                .cpu()
                .numpy()
                .astype(int)
            )
            x1 = int(np.clip(x1, 0, width - 1))
            x2 = int(np.clip(x2, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            y2 = int(np.clip(y2, 0, height - 1))

            if x2 > x1 and y2 > y1:
                cv2.rectangle(
                    original_mask,
                    (x1, y1),
                    (x2, y2),
                    1,
                    thickness=-1,
                )

        display_mask = (
            cv2.resize(
                original_mask,
                (display_w, display_h),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        ).astype(np.uint8)

        return {
            "detection_available": True,
            "label": label,
            "prediction": prediction,
            "signed_score": signed_score,
            "threshold": threshold,
            "benign_score": p_b,
            "malignant_score": p_m,
            "top_detection_class": top_class_name,
            "top_detection_confidence": top_confidence,
            "mask": display_mask,
        }

    def _generate_cams(
        self,
        image_tensor,
        predicted_class: int,
        display_shape,
    ):
        padded_maps = logic.generate_frozen_cam_maps(
            wrapper=self.classifier_wrapper,
            image_tensor=image_tensor,
            predicted_class=predicted_class,
            xai_configs=self.xai_configs,
        )

        display_h = int(display_shape[0])
        display_w = int(display_shape[1])

        return {
            method_name: logic.unpad_square_to_shape(
                raw_map,
                target_height=display_h,
                target_width=display_w,
                interpolation=cv2.INTER_LINEAR,
            )
            for method_name, raw_map
            in padded_maps.items()
        }

    def _create_report(
        self,
        image_id,
        classifier,
        yolo,
        cam_maps,
    ):
        if yolo["detection_available"]:
            diagnostic_agreement = (
                classifier["prediction"]
                == yolo["prediction"]
            )
        else:
            diagnostic_agreement = None

        if yolo["detection_available"]:
            layercam = np.asarray(
                cam_maps[
                    "Layer-CAM"
                ],
                dtype=np.float32,
            )

            yolo_mask = (
                np.asarray(
                    yolo[
                        "mask"
                    ]
                )
                > 0
            )

            if (
                layercam.ndim != 2
                or yolo_mask.ndim != 2
                or layercam.shape != yolo_mask.shape
                or not yolo_mask.any()
                or not np.isfinite(layercam).any()
            ):
                location_agreement = None
            else:
                safe_cam = np.where(
                    np.isfinite(layercam),
                    layercam,
                    -np.inf,
                )

                peak_flat = int(
                    np.argmax(
                        safe_cam
                    )
                )

                peak_y, peak_x = np.unravel_index(
                    peak_flat,
                    safe_cam.shape,
                )

                location_agreement = bool(
                    yolo_mask[
                        peak_y,
                        peak_x,
                    ]
                )
        else:
            location_agreement = None

        if not yolo["detection_available"]:
            evidence_summary = {
                "text": "Incomplete spatial evidence: YOLO did not localise a nodule",
                "color": "#A66A00",
            }
        elif (
            diagnostic_agreement
            and location_agreement
        ):
            evidence_summary = {
                "text": f"Concordant {classifier['label'].lower()} model outputs",
                "color": "#1B7F3A",
            }
        elif diagnostic_agreement:
            evidence_summary = {
                "text": (
                    "Diagnostic agreement with spatially discordant "
                    "Layer-CAM focus"
                ),
                "color": "#A66A00",
            }
        else:
            evidence_summary = {
                "text": "Discordant benign/malignant model outputs",
                "color": "#B33A3A",
            }

        return {
            "image_id": image_id,

            "cnn_label": classifier["label"],
            "cnn_malignancy_score": classifier["malignancy_score"],
            "cnn_threshold": classifier["threshold"],
            "cnn_mask_fraction": classifier["mask_fraction"],

            "yolo_detection_available": yolo["detection_available"],
            "yolo_label": yolo["label"],
            "yolo_signed_score": yolo["signed_score"],
            "yolo_threshold": yolo["threshold"],
            "yolo_top_detection_class": yolo["top_detection_class"],
            "yolo_top_detection_confidence": (
                yolo["top_detection_confidence"]
            ),

            "diagnostic_agreement": diagnostic_agreement,
            "location_agreement": location_agreement,
            "location_agreement_definition": (
                "maximum Layer-CAM activation lies inside the YOLO-predicted "
                "nodule mask"
            ),
            "evidence_summary": evidence_summary,
        }

    @staticmethod
    def _align_map_to_display(
        spatial_map,
        display_rgb,
        *,
        is_mask: bool,
        name: str,
    ):
        target_h, target_w = map(int, display_rgb.shape[:2])
        arr = np.asarray(spatial_map)

        if arr.ndim > 2:
            arr = np.squeeze(arr)

        if arr.ndim != 2:
            raise RuntimeError(
                f"{name} must be a 2-D spatial map; got shape {arr.shape}"
            )

        if tuple(arr.shape) == (target_h, target_w):
            aligned = arr
        elif arr.shape[0] == arr.shape[1]:

            aligned = logic.unpad_square_to_shape(
                arr,
                target_height=target_h,
                target_width=target_w,
                interpolation=(
                    cv2.INTER_NEAREST
                    if is_mask
                    else cv2.INTER_LINEAR
                ),
            )
        else:
            aligned = cv2.resize(
                arr,
                (target_w, target_h),
                interpolation=(
                    cv2.INTER_NEAREST
                    if is_mask
                    else cv2.INTER_LINEAR
                ),
            )

        if is_mask:
            aligned = (aligned > 0).astype(np.uint8)
        else:
            aligned = logic.norm01(
                np.asarray(aligned, dtype=np.float32)
            )

        if tuple(aligned.shape[:2]) != (target_h, target_w):
            raise RuntimeError(
                f"{name} could not be aligned to display geometry: "
                f"got {aligned.shape[:2]}, expected {(target_h, target_w)}"
            )

        return aligned

    @staticmethod
    def _safe_mask_overlay(
        display_rgb,
        mask,
        alpha: float = 0.42,
    ):
        image = np.asarray(display_rgb).copy()
        target_h, target_w = image.shape[:2]

        mask_bool = np.asarray(mask).astype(bool)
        if tuple(mask_bool.shape[:2]) != (target_h, target_w):
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8),
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        if not mask_bool.any():
            return image

        overlay = image.astype(np.float32)
        overlay[mask_bool] = (
            (1.0 - alpha) * overlay[mask_bool]
            + alpha * np.asarray(
                [255.0, 255.0, 255.0],
                dtype=np.float32,
            )
        )

        output = np.clip(
            overlay,
            0,
            255,
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        bgr = cv2.cvtColor(
            output,
            cv2.COLOR_RGB2BGR,
        )
        cv2.drawContours(
            bgr,
            contours,
            -1,
            (255, 255, 255),
            2,
        )

        return cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

    @staticmethod
    def _safe_cam_overlay(
        display_rgb,
        cam,
        alpha: float = 0.45,
    ):
        image = np.asarray(display_rgb)
        target_h, target_w = image.shape[:2]

        cam = np.asarray(cam, dtype=np.float32)
        if tuple(cam.shape[:2]) != (target_h, target_w):
            cam = cv2.resize(
                cam,
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )

        cam = logic.norm01(cam)

        heat = cv2.applyColorMap(
            np.uint8(np.clip(cam, 0.0, 1.0) * 255.0),
            cv2.COLORMAP_JET,
        )
        heat = cv2.cvtColor(
            heat,
            cv2.COLOR_BGR2RGB,
        )

        return cv2.addWeighted(
            image,
            1.0 - alpha,
            heat,
            alpha,
            0.0,
        )

    @staticmethod
    def _prepare_plot_data(
        display_rgb,
        classifier,
        yolo,
        cam_maps,
    ):
        display_shape = tuple(display_rgb.shape[:2])

        classifier_mask = App._align_map_to_display(
            classifier["mask"],
            display_rgb,
            is_mask=True,
            name="CNN segmentation mask",
        )

        yolo_mask = App._align_map_to_display(
            yolo["mask"],
            display_rgb,
            is_mask=True,
            name="YOLO mask",
        )

        aligned_cams = {
            name: App._align_map_to_display(
                cam,
                display_rgb,
                is_mask=False,
                name=name,
            )
            for name, cam in cam_maps.items()
        }

        yolo_overlay = App._safe_mask_overlay(
            display_rgb,
            yolo_mask,
        )

        cnn_seg_overlay = App._safe_mask_overlay(
            display_rgb,
            classifier_mask,
        )

        cam_overlays = {
            name: App._safe_cam_overlay(
                display_rgb,
                cam,
            )
            for name, cam in aligned_cams.items()
        }

        classifier["mask"] = classifier_mask
        yolo["mask"] = yolo_mask
        cam_maps.clear()
        cam_maps.update(aligned_cams)

        return {
            "display_rgb": display_rgb,
            "yolo_overlay": yolo_overlay,
            "cnn_seg_overlay": cnn_seg_overlay,
            "cam_overlays": cam_overlays,
        }

def main():

    classifier_path = (
        Path(CLASSIFIER_CHECKPOINT).expanduser().resolve()
        if CLASSIFIER_CHECKPOINT is not None
        else _find_mobilenet_checkpoint()
    )

    yolo_path = (
        Path(YOLO_CHECKPOINT).expanduser().resolve()
        if YOLO_CHECKPOINT is not None
        else _find_yolo_checkpoint()
    )

    classifier, checkpoint = (
        _load_mobilenet_classifier(
            classifier_path
        )
    )

    classifier_wrapper = (
        logic.ClassificationOnlyWrapper(classifier)
        .to(logic.DEVICE)
        .eval()
    )

    xai_configs, xai_freeze_path = (
        logic.resolve_xai_layers(
            PROJECT_ROOT,
            classifier_wrapper,
        )
    )

    yolo_freeze, yolo_freeze_path = (
        logic.load_yolo_evaluation_freeze(
            PROJECT_ROOT
        )
    )

    if yolo_freeze is None:

        yolo_config = {
            "image_threshold": 0.0,
            "augment": False,
            "source": "development argmax-equivalent fallback",
        }
    else:
        yolo_config = {
            "image_threshold": float(
                yolo_freeze["image_threshold"]
            ),
            "augment": bool(
                yolo_freeze.get("augment", False)
            ),
            "source": "validation-frozen YOLO evaluation",
        }

    yolo = YOLO(str(yolo_path))

    if yolo_freeze_path is None:
        print("Warning: yolo_evaluation_freeze.json not found; using threshold 0.0.")

    if xai_freeze_path is None:
        print("Warning: protocol_B_freeze.json not found; using fallback CAM layers.")

    root = tkinter.Tk()

    App(
        root=root,
        classifier=classifier,
        classifier_checkpoint=checkpoint,
        classifier_wrapper=classifier_wrapper,
        yolo=yolo,
        xai_configs=xai_configs,
        xai_freeze_path=xai_freeze_path,
        yolo_config=yolo_config,
        yolo_freeze_path=yolo_freeze_path,
    )

    root.mainloop()

if __name__ == "__main__":
    main()
