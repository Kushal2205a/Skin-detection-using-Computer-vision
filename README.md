# Skin Detection using Computer Vision

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/github/license/Kushal2205a/Skin-detection-using-Computer-vision)](./LICENSE)

Trained a ResNet50-based classifier on skin image classes and evaluate robustness across different brightness levels.

---

## Features

- Dataset loading via `torchvision.datasets.ImageFolder`
- Training + validation split
- Checkpoint save/load (`best_model.pth`)
- Brightness robustness evaluation + JSON export (`brightness_results.json`)
- Plots saved as `.png` files for quick review

---

## Results (repo images)

### Robustness to brightness
![Robustness to brightness](./brightness_robustness.png)

### Most sensitive class vs brightness
![Sensitive class](./sensitive_class.png)

### Sensitivity across all classes
![Sensitivity (all classes)](./sensitivity_all_classes.png)

---

## Setup

Create and activate a virtual environment :

```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
```

Install dependencies:

```bash
pip install torch torchvision matplotlib numpy pillow
```

---

## Run

```bash
python skin_disease.py
```

### Outputs

- `best_model.pth`
- `brightness_results.json`
- `brightness_robustness.png`
- `sensitive_class.png`
- `sensitivity_all_classes.png`

---

## Notes

- Uses GPU automatically if available.
- Validation split is created from the training set inside the script.

---


