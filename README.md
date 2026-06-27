# PASTIS Dataset for Crop Segmentation | Sentinel-2 | Prithvi EO 2.0 | PyTorch | TerraTorch

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-darkblue.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Sentinel-2](https://img.shields.io/badge/Satellite-Sentinel--2-green.svg)](https://copernicus.eu/)
[![Remote Sensing](https://img.shields.io/badge/Domain-Remote%20Sensing-darkgreen.svg)](https://github.com/)
[![Semantic Segmentation](https://img.shields.io/badge/Task-Semantic%20Segmentation-orange.svg)](https://github.com/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TerraTorch](https://img.shields.io/badge/Library-TerraTorch-purple.svg)](https://github.com/)
[![Dataset](https://img.shields.io/badge/Data-PASTIS%20Dataset-lightgrey.svg)](https://github.com/)
[![GitHub stars](https://img.shields.io/github/stars/username/repo.svg?style=social)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🛰️ Project Description

Welcome to the **PASTIS Dataset for Crop Segmentation** repository. This repository provides a streamlined, deep learning-ready version of the benchmark PASTIS dataset, optimized specifically for **semantic segmentation, crop classification, and parcel segmentation** tasks using high-resolution **Sentinel-2 satellite imagery**.

### Why This Repository Exists
As Earth Observation (EO) advances into the era of GeoAI, there is a critical need for standardized, highly accessible data formats to benchmark advanced deep learning models. This repository bridges the gap between raw remote sensing data and modern AI frameworks. By structuring the **PASTIS dataset** into clean, pre-processed NumPy matrices (`.npy`), this project offers an out-of-the-box solution for training state-of-the-art vision models, fine-tuning modern **Earth Observation Foundation Models** (such as IBM/NASA's **Prithvi EO 2.0**), and experimenting with geospatial packages like **TerraTorch** and **TorchGeo**.

### Intended Research Usage
This curated **satellite image segmentation dataset** is designed for researchers and developers working in Agricultural AI, Remote Sensing, and Machine Learning. Whether you are benchmarking a classic U-Net in **PyTorch**, configuring a **Vision Transformer (ViT)**, or fine-tuning massive geospatial foundation models, this repository provides a lightweight, frictionless entry point to high-quality satellite training data.

---

## ✨ Features

* **⚡ Optimized Data Format:** All data is stored in standard NumPy (`.npy`) format for high-speed, memory-mapped I/O during training loops.
* **📊 Diverse Layout:** Includes **500 core training patches** alongside a dedicated subset of **40 independent inference samples** for validation and model testing.
* **🧩 Framework Ready:** Fully compatible with mainstream ecosystems including **PyTorch, TorchGeo, TerraTorch, and Hugging Face**.
* **🌍 Foundation Model Friendly:** Designed out-of-the-box for fine-tuning advanced geospatial foundation models such as **Prithvi EO 2.0** and **TerraTorch architectures**.
* **📦 Zero-Friction Setup:** Simplified repository structure makes downloading, visualizing, and training rapid and straightforward.

---

## 📂 Folder Structure

The repository maintains a clean, intuitive structure tailored for immediate integration with custom PyTorch `Dataset` loaders:

```text
PASTIS/
│
├── images/
│     └── *.npy          # 500 Sentinel-2 image patches (Training/Validation)
│
├── masks/
│     └── *.npy          # 500 Semantic crop segmentation masks (Training/Validation)
│
├── inference/
│     ├── images/
│     │      └── *.npy   # 40 Independent inference image patches
│     │
│     └── masks/
│            └── *.npy   # 40 Ground-truth inference masks for testing
│
└── README.md

```

---

## 📊 Dataset Details

| Folder Path | File Count | File Format | Spatial Domain / Target | Description |
| --- | --- | --- | --- | --- |
| `images/` | 500 | `.npy` | Sentinel-2 Satellite Patches | Multispectral image patches optimized for training. |
| `masks/` | 500 | `.npy` | Pixel-level Crop Categories | Explicit semantic labels corresponding to parcel boundaries and crop classes. |
| `inference/images/` | 40 | `.npy` | Hold-out Evaluation Images | Unseen evaluation patches used to assess model generalization. |
| `inference/masks/` | 40 | `.npy` | Evaluation Ground-Truth | Ground-truth masks reserved for calculating accuracy metrics (mIoU, F1-score). |

---

## 🚀 Quick Start

Get your **satellite image segmentation** pipeline up and running in minutes.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/PASTIS-Crop-Segmentation.git
cd PASTIS-Crop-Segmentation

```

### 2. Install Dependencies

Ensure you have the required Python libraries installed:

```bash
pip install numpy torch matplotlib

```

### 3. Load and Visualize via Python

Use the following standardized Python sample code to quickly load, inspect, and verify the shapes of your **PASTIS dataset** array arrays:

```python
import os
import numpy as np
import matplotlib.pyplot as plt

# Define relative paths
IMAGE_PATH = 'images/patch_001.npy'
MASK_PATH = 'masks/patch_001.npy'

# Verify files exist before loading
if os.path.exists(IMAGE_PATH) and os.path.exists(MASK_PATH):
    # Load NumPy arrays
    image = np.load(IMAGE_PATH)
    mask = np.load(MASK_PATH)
    
    # Print shapes for framework verification (e.g., PyTorch/TerraTorch compatibility)
    print(f"🛰️ Loaded Sentinel-2 Image Shape: {image.shape}")
    print(f"🧩 Loaded Segmentation Mask Shape: {mask.shape}")
    
    # Quick visualization snippet (Using standard RGB channels or false color)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image[..., :3] if image.ndim == 3 else image)
    ax[0].set_title("Sentinel-2 Patch")
    ax[0].axis('off')
    
    ax[1].imshow(mask, cmap='jet')
    ax[1].set_title("Crop Mask")
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Sample files not found. Please verify your working directory paths.")

```

---

## 🔬 Research Applications

This **remotely sensed deep learning dataset** is purpose-built to accelerate discoveries across several domain fields:

* 🌾 **Crop Mapping & Type Classification:** High-fidelity field parcel segmentation based on multi-temporal spectral profiles.
* 🛰️ **Earth Observation & GeoAI:** Benchmarking state-of-the-art architectures against challenging land-cover distributions.
* 🚜 **Precision Agriculture:** Delineating accurate boundaries for automated tractor guidance and yield prediction.
* 🌲 **Land Cover Mapping & Change Detection:** Assessing land-use alterations, deforestation, or agricultural urbanization impacts over time.
* 🤖 **Geospatial Foundation Models:** Training, evaluation, and fine-tuning scenarios involving next-generation spatial models like **Prithvi EO 2.0** or **Vision Transformers (ViTs)**.

---

## 🔌 Compatible Frameworks

This variant of the **PASTIS dataset** is carefully configured to slide effortlessly into modern deep learning suites:

* **PyTorch / TorchVision:** Write standard `torch.utils.data.Dataset` pipelines wrapping these pre-processed arrays.
* **TerraTorch & TorchGeo:** Directly ingest patches into specialized pipelines curated for geospatial model benchmarking.
* **Hugging Face Transformers:** Map multispectral configurations into pipeline architectures designed for Vision Transformers (`ViTForSemanticSegmentation`).
* **IBM NASA Prithvi EO 2.0:** Adapt downstream fine-tuning scripts to utilize these patches for targeted agricultural classification downstream evaluations.

---

## 🤝 Contributing

Contributions to enhance this loader repository are welcome! If you want to contribute (e.g., provide pre-built PyTorch training files, add visualization tutorials, or optimize loading scripts), feel free to follow these steps:

1. Fork the project repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 🏆 Acknowledgements

We express our gratitude to the creators of the original **PASTIS Dataset** for providing the foundational benchmark materials. Additional credit goes to the **European Space Agency (ESA) Copernicus** program for the accessible Sentinel-2 open data architecture, alongside **IBM** and **NASA** for pushing the boundaries of Earth Observation via open frameworks like **Prithvi EO 2.0** and **TerraTorch**.

---

## 🔍 Search Discovery Keywords

```text
PASTIS Dataset, Sentinel-2 Dataset, Crop Segmentation Dataset, Satellite Dataset, Semantic Segmentation Dataset, Remote Sensing Dataset, Earth Observation Dataset, GeoAI Dataset, Agriculture AI, Agricultural Dataset, Land Cover Dataset, Parcel Segmentation, Prithvi EO 2.0, IBM NASA Prithvi, TerraTorch, PyTorch Dataset, NumPy Dataset, Satellite Image Dataset, Deep Learning Dataset, Geospatial Dataset, Foundation Model Dataset, Vision Transformer Dataset, Satellite Segmentation, Crop Classification, Agricultural Monitoring, Remote Sensing AI, Earth Observation AI, GeoSpatial Deep Learning, Satellite Imagery, AI Dataset, Machine Learning Dataset.

```

```
