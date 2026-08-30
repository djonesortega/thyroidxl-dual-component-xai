# Supplementary Index

This document describes the supplementary materials developed, how to install and run the project locally, and where to obtain the required ThyroidXL dataset.

---

## Install Guide

1. **Clone or download the full project from the GitHub repository**

   * GitHub: **https://github.com/djonesortega/thyroidxl-dual-component-xai**

   * The trained `.pt` model checkpoints are stored using Git LFS. If cloning the repository, please make sure Git LFS is installed and run:

```bash
git lfs install
git lfs pull
```

2. **Download the ThyroidXL dataset**

   * ThyroidXL Hugging Face: **https://huggingface.co/datasets/hunglc007/ThyroidXL**

   * The ThyroidXL dataset is not redistributed directly in this repository.

3. **Please use Python 3.11**

4. **To install the requirements please run:**

```bash
pip install -r requirements.txt
```

---

## Links

1. **ThyroidXL dataset**: https://huggingface.co/datasets/hunglc007/ThyroidXL

2. **Full project GitHub repository**: https://github.com/djonesortega/thyroidxl-dual-component-xai

---

## Project Content Description

### 1) Dataset

* The project uses the publicly available **ThyroidXL** thyroid ultrasound dataset.

* The original dataset can be downloaded from Hugging Face and should be made available locally before reproducing training or evaluation.

* The dataset itself is not included in this GitHub repository.

---

### 2) Dual Model Pipeline

* `logic.py`: preprocessing, model-loading, inference, segmentation, and XAI utilities used by the dual-model visualisation.

* `dual_model_pipeline.py`: core visualisation for the dual-component system, combining CNN classification/XAI with YOLOv8s-seg localisation and segmentation.

To run the visualisation:

```bash
python dual_model_pipeline.py
```

---

### 3) Evaluation

The evaluation code contains the analyses used for the publication.

* **Individual model evaluation**

  * MobileNetV3-Large
  * EfficientNet-B3
  * ConvNeXt-Tiny
  * YOLOv8s-seg

* **Quantitative XAI evaluation**

  * Grad-CAM
  * Grad-CAM++
  * LayerCAM

* **Cross-model statistical analysis**

  * Comparison of the four final models on the official held-out cohort.

* **Nodule-size subgroup analysis**

  * Classification and segmentation performance across training-defined small, medium, and large nodule groups.

* **CNN–YOLO diagnostic agreement**

  * ConvNeXt-Tiny and YOLOv8s-seg patient-level agreement analysis.

* **CNN–YOLO spatial agreement**

  * MobileNetV3-Large LayerCAM and YOLOv8s-seg spatial agreement analysis.

---

### 4) Results

The `results` folder contains the saved outputs from the final publication analyses.

These include:

* image-level predictions
* patient-level predictions
* model evaluation summaries
* ROC and precision-recall outputs
* calibration outputs for the CNN models
* segmentation results
* quantitative XAI results
* four-model statistical comparisons
* nodule-size subgroup results
* ConvNeXt-Tiny–YOLO diagnostic agreement
* MobileNetV3-Large LayerCAM–YOLO spatial agreement

The saved result files are included so that the reported outputs can be inspected without rerunning every evaluation.

---

### 5) Models

The `Models` folder contains the trained model checkpoints used for the final evaluation:

* **MobileNetV3-Large**
* **EfficientNet-B3**
* **ConvNeXt-Tiny**
* **YOLOv8s-seg**

The `.pt` files are stored using Git LFS.

---

### 6) Training

* **Google Colab Training**: `.ipynb` notebooks for all four models, including the saved training outputs.

* The notebooks contain the training configurations used for:

  * MobileNetV3-Large
  * EfficientNet-B3
  * ConvNeXt-Tiny
  * YOLOv8s-seg

* The CNN models perform malignancy classification together with auxiliary nodule segmentation.

* YOLOv8s-seg is trained independently for nodule localisation, segmentation, and benign/malignant class assignment.

---

### 7) requirements.txt

* Python requirements required to reproduce the environment and run the code locally.

* Please use Python 3.11.

* To install the requirements:

```bash
pip install -r requirements.txt
```

---

## Citation

### ThyroidXL Dataset

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

---

