# 🧠 Fingerprint Recognition Using EfficientNet-B0

## Overview

This project implements a **fingerprint recognition system** using **PyTorch** and **transfer learning** with **EfficientNet-B0**. The model is trained to classify fingerprints based on individual students’ roll numbers.

The pipeline includes:

* Data augmentation to increase dataset size
* Multi-GPU training support
* Early stopping and learning rate scheduling
* Reproducible training with fixed random seeds
* Visualization of training metrics and sample images

---

## Dataset

* The dataset folder structure is as follows:

```
Fingerprint Dataset/
│
├── 1/          # Class 1 (Student Roll No. 1)
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── 2/          # Class 2 (Student Roll No. 2)
│   ├── img1.jpg
│   └── ...
└── ...
```

* Each class folder contains multiple fingerprint images.
* The dataset can be augmented to expand the number of samples for better model generalization.

---

## Features

1. **Data Augmentation**:

   * Random rotations, flips, perspective transforms, sharpness adjustment, brightness/contrast changes, etc.
   * Augmentation increases dataset size by 10× per image.

2. **Model**:

   * Pretrained **EfficientNet-B0**
   * Feature extractor frozen for transfer learning
   * Classifier modified with dropout for regularization

3. **Training Enhancements**:

   * Multi-GPU support
   * Early stopping with patience
   * Learning rate scheduler (ReduceLROnPlateau)
   * Label smoothing for better generalization

4. **Reproducibility**:

   * Fixed seed ensures deterministic results

5. **Visualization**:

   * Training & validation loss curves
   * Validation accuracy curves
   * Sample fingerprint images with class labels

---

## Installation

Make sure you have Python 3.8+ and PyTorch installed. Recommended environment:

```bash
pip install torch torchvision matplotlib numpy pillow scikit-learn tqdm
```

---

## Usage

1. **Update the dataset path** in the code:

```python
data_dir = "/path/to/fingerprint-dataset"
```

2. **Run the training notebook or script**. The model will automatically:

   * Augment the data
   * Train with EfficientNet-B0
   * Save the best model as `best_fingerprint_model_final.pth`

3. **Visualize results** using the included visualization section.

---

## Outputs

* **Trained model**: `best_fingerprint_model_final.pth`
* **Training curves**: loss and accuracy plots
* **Sample images**: random fingerprint images from the dataset

---

## Notes

* The current code is optimized for **small datasets** (few images per class).
* For larger datasets, adjust `batch_size` and `num_workers` accordingly.
* Early stopping and learning rate scheduling help prevent overfitting.

---

## References

* EfficientNet: [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)
* PyTorch Transfer Learning Tutorial: [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
