# ThyroidXL Dual-Component XAI

Code, trained models, evaluation outputs, and supplementary material for:

**Explainable Dual-Component Deep Learning for Thyroid Nodule Malignancy Assessment on Ultrasound**

This repository contains the implementation used to evaluate complementary deep-learning approaches for thyroid nodule malignancy classification, nodule localisation and segmentation, quantitative explainability, subgroup analysis, and CNN–YOLO agreement using the public **ThyroidXL** dataset.

---

## Dataset

The ThyroidXL dataset is publicly available from Hugging Face:

**https://huggingface.co/datasets/hunglc007/ThyroidXL**

The dataset itself is not redistributed in this repository.

---

## Installation

Python 3.11 is recommended.

Clone the repository:

```bash
git clone https://github.com/djonesortega/thyroidxl-dual-component-xai.git
cd thyroidxl-dual-component-xai
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### Git LFS

Trained PyTorch checkpoints (`.pt`) are stored using Git LFS.

Install and initialise Git LFS before pulling the model files:

```bash
git lfs install
git lfs pull
```

---

## Repository Structure

```text
thyroidxl-dual-component-xai/
│
├── Models/
│   ├── MobileNetV3/
│   ├── EfficientNetB3/
│   ├── ConvNeXtTiny/
│   └── YOLOv8sSeg/
│
├── Training/
│   ├── MobileNetV3/
│   ├── EfficientNetB3/
│   ├── ConvNeXtTiny/
│   └── YOLOv8sSeg/
│
├── Evaluation/
│   ├── mobilenetv3_xai_evaluation.py
│   ├── efficientnetb3_xai_evaluation.py
│   ├── convnexttiny_xai_evaluation.py
│   ├── yolo_publication_evaluation.py
│   ├── freeze_selected_cnn_before_test.py
│   ├── cross_model_publication_statistics_4models.py
│   ├── component_subgroup_publication_analysis_4models.py
│   ├── convnext_yolo_diagnostic_selective.py
│   └── mobilenet_layercam_yolo_spatial.py
│
├── results/
│   ├── MobileNet/
│   ├── EfficientNetB3/
│   ├── ConvNeXtTiny/
│   ├── YOLO/
│   └── Combined/
│
├── dual_model_pipeline.py
├── logic.py
├── requirements.txt
└── README.md
```

---

## Models

Four deep-learning architectures are evaluated:

- **MobileNetV3-Large** — malignancy classification with auxiliary nodule segmentation.
- **EfficientNet-B3** — malignancy classification with auxiliary nodule segmentation.
- **ConvNeXt-Tiny** — malignancy classification with auxiliary nodule segmentation.
- **YOLOv8s-seg** — independent nodule localisation, segmentation, and benign/malignant class assignment.

The CNN models use ImageNet-pretrained backbones and a multitask classification/segmentation architecture. YOLOv8s-seg is trained independently as the second component of the framework.

---

## Training

The `Training/` directory contains the notebooks used to train the four models.

The notebooks retain the training outputs used in the study and include the configuration required to reproduce the model-training procedure.

Model and epoch selection were performed using a patient-disjoint development partition before final refitting on the complete official training cohort.

---

## Evaluation

The `Evaluation/` directory contains the scripts used for the final publication analyses.

### Individual Model Evaluation

The model evaluation pipelines produce:

- image-level predictions
- patient-level predictions
- ROC-AUC and average precision
- accuracy, sensitivity, specificity, precision, F1-score, and MCC
- bootstrap confidence intervals
- segmentation Dice and IoU
- calibration analysis for the CNN models

### Quantitative XAI

Grad-CAM, Grad-CAM++, and LayerCAM are evaluated against expert nodule annotations.

Reported XAI measures include:

- Top-15% activation IoU
- overlap-hit rate
- pointing-hit rate
- nodule energy fraction

Activation maps are mapped back to the original ultrasound geometry before spatial evaluation.

### Four-Model Statistical Analysis

The final held-out predictions from MobileNetV3-Large, EfficientNet-B3, ConvNeXt-Tiny, and YOLOv8s-seg are compared using paired statistical analyses.

### Nodule-Size Subgroup Analysis

Nodule-size boundaries are derived exclusively from the official training cohort and then applied unchanged to the held-out cohort.

Classification and segmentation performance are evaluated across small, medium, and large nodule groups.

### CNN–YOLO Diagnostic Agreement

ConvNeXt-Tiny and YOLOv8s-seg are compared at the patient level to determine the proportion of cases for which both independently trained components produce the same benign/malignant prediction.

### CNN–YOLO Spatial Agreement

MobileNetV3-Large LayerCAM and YOLOv8s-seg masks are compared after mapping both outputs to the original ultrasound image geometry.

---

## Dual-Component Visualisation

The repository also includes an interactive dual-component visualisation:

- `dual_model_pipeline.py` — main application
- `logic.py` — model loading, preprocessing, inference, and XAI utilities

Run:

```bash
python dual_model_pipeline.py
```

---

## Results

The `results/` directory contains the final saved outputs used for the publication analyses, including:

- patient- and image-level predictions
- quantitative XAI measurements
- model-performance summaries
- segmentation results
- nodule-size subgroup analyses
- four-model statistical comparisons
- CNN–YOLO diagnostic agreement
- CNN–YOLO spatial agreement

These files are provided to make the reported analyses transparent and reproducible without requiring every evaluation to be rerun.

---

## Citation

### ThyroidXL Dataset

If you use the ThyroidXL dataset in your research, please cite:

```bibtex
@inproceedings{10.1007/978-3-032-05182-0_60,
  author = {Duong, Viet Hung and Vu, Huan and Phan, Huong Duong and Nguyen, Duc Quyen and Pham, Duc Hao and Le, Quang Toan and Nguyen, Ba Sy and Do, Tien Dung and Dinh, Viet Sang and Nguyen, Tien Cuong and Pham, Huy Hoang and Ngo, Dien Hy},
  title = {ThyroidXL: Advancing Thyroid Nodule Diagnosis with an Expert-Labeled, Pathology-Validated Dataset},
  year = {2025},
  isbn = {978-3-032-05181-3},
  publisher = {Springer-Verlag},
  address = {Berlin, Heidelberg},
  url = {https://doi.org/10.1007/978-3-032-05182-0_60},
  doi = {10.1007/978-3-032-05182-0_60}
}
```

### This Work

If you use the code, trained models, or results from this repository, please cite the associated publication:

**Daniel Jones Ortega and Yasir Hafeez. _Explainable Dual-Component Deep Learning for Thyroid Nodule Malignancy Assessment on Ultrasound._**

The final journal citation and DOI can be added here once available.

---

## Acknowledgements

We thank the creators and contributors of the ThyroidXL dataset for making the dataset publicly available.

---

## License

Please refer to the repository license for the source code. Use of the ThyroidXL dataset remains subject to the terms and conditions of the original dataset provider.
