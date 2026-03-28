# Multi-Model Unified Organ Cancer Identifier

A deep learning framework for automated cancer detection across multiple organs using 3D convolutional neural networks. The system processes volumetric CT scans and classifies regions of interest as malignant or benign, with interpretable Grad-CAM visualizations for clinical transparency.

This project accompanies a research paper investigating whether lightweight 3D architectures can achieve high diagnostic accuracy on both consumer-grade and cloud GPUs, making advanced cancer screening more accessible.

---

## Architecture Overview

Two compact 3D CNNs are trained and compared on each organ:

| Model | Parameters | Design Philosophy |
|-------|-----------|-------------------|
| **ResNet3D** | 3.62M | Residual connections with bottleneck blocks for gradient stability |
| **VGG3D** | 0.87M | Sequential convolutional stack prioritizing simplicity and speed |

Both architectures are organ-agnostic. The same model definitions are reused across all organs with no structural modifications, isolating the effect of data characteristics on performance.

### Shared Training Configuration

- **Validation:** Patient-level 3-fold cross-validation
- **Loss:** Focal Loss (gamma=2, alpha=0.25) for class imbalance
- **Optimizer:** AdamW with cosine annealing + linear warmup
- **Regularization:** Dropout, weight decay, early stopping (patience=25)
- **Augmentation:** Random 3D flips, 90-degree rotations, Gaussian noise, intensity shifts
- **Efficiency:** Mixed-precision (AMP), gradient accumulation (effective batch size=16)
- **Inference:** 7-transform test-time augmentation with prediction averaging

---

## Organs and Results

### Lung Cancer Detection — Complete

**Dataset:** [LUNA16](https://luna16.grand-challenge.org/) (888 CT scans, 1,186 annotated nodules)

**Preprocessing:** Raw `.mhd` volumes resampled to isotropic 1 mm spacing, HU-windowed to [-1000, 400], cropped into 64x64x64 patches centered on candidate coordinates. Positive/negative sampling balanced via `WeightedRandomSampler`.

<table>
<tr>
<th>Model</th><th>AUC-ROC</th><th>Sensitivity</th><th>Specificity</th><th>F1 Score</th><th>Accuracy</th>
</tr>
<tr>
<td><b>ResNet3D</b></td><td>0.9918 +/- 0.0059</td><td>0.9074 +/- 0.0367</td><td>0.9816 +/- 0.0029</td><td>0.9076 +/- 0.0193</td><td>0.9692 +/- 0.0063</td>
</tr>
<tr>
<td><b>VGG3D</b></td><td>0.9944 +/- 0.0021</td><td>0.9243 +/- 0.0192</td><td>0.9831 +/- 0.0017</td><td>0.9203 +/- 0.0095</td><td>0.9733 +/- 0.0032</td>
</tr>
</table>

*All values reported as mean +/- standard deviation across 3 folds (8,106 test samples).*

**Key finding:** VGG3D (0.87M params) outperformed ResNet3D (3.62M params) across every metric with lower variance, suggesting that smaller models generalize better when medical imaging data is limited.

#### ROC Curves

<p align="center">
<img src="results/lung/plots/comparison_roc.png" width="600" alt="Comparison ROC curves for lung nodule classification">
</p>

#### Training Curves

<p align="center">
<img src="results/lung/plots/resnet3d/training_curves_kfold.png" width="48%" alt="ResNet3D training curves">
<img src="results/lung/plots/vgg3d/training_curves_kfold.png" width="48%" alt="VGG3D training curves">
</p>

#### Grad-CAM Visualizations

3D Grad-CAM heatmaps overlaid on axial CT slices, showing model attention for true positives, true negatives, false positives, and false negatives.

**VGG3D:**
<p align="center">
<img src="results/lung/plots/gradcam/vgg3d/tp_1.png" width="24%">
<img src="results/lung/plots/gradcam/vgg3d/tn_1.png" width="24%">
<img src="results/lung/plots/gradcam/vgg3d/fp_1.png" width="24%">
<img src="results/lung/plots/gradcam/vgg3d/fn_1.png" width="24%">
</p>
<p align="center"><i>Left to right: True Positive, True Negative, False Positive, False Negative</i></p>

**ResNet3D:**
<p align="center">
<img src="results/lung/plots/gradcam/resnet3d/tp_1.png" width="24%">
<img src="results/lung/plots/gradcam/resnet3d/tn_1.png" width="24%">
<img src="results/lung/plots/gradcam/resnet3d/fp_1.png" width="24%">
<img src="results/lung/plots/gradcam/resnet3d/fn_1.png" width="24%">
</p>
<p align="center"><i>Left to right: True Positive, True Negative, False Positive, False Negative</i></p>

---

### Liver Cancer Detection — Complete

**Dataset:** [LiTS — Liver Tumor Segmentation Challenge](https://competitions.codalab.org/competitions/17094) (131 contrast-enhanced abdominal CT scans with tumor masks). For pipeline compatibility, files are organized in `data/Task03_Liver/` format (`imagesTr` and `labelsTr`).

**Preprocessing:** NIfTI volumes were resampled to 1 mm isotropic, HU-windowed to [-200, 300], and converted into 96x96x96 patches. Positive patches were centered on tumor components; negatives were sampled from tumor-free liver regions.

<table>
<tr>
<th>Model</th><th>AUC-ROC</th><th>Sensitivity</th><th>Specificity</th><th>F1 Score</th><th>Accuracy</th>
</tr>
<tr>
<td><b>ResNet3D</b></td><td>0.8297 +/- 0.1100</td><td>0.6899 +/- 0.1434</td><td>0.8526 +/- 0.0577</td><td>0.6846 +/- 0.1251</td><td>0.8019 +/- 0.0791</td>
</tr>
<tr>
<td><b>VGG3D</b></td><td>0.9464 +/- 0.0105</td><td>0.8456 +/- 0.0231</td><td>0.9444 +/- 0.0090</td><td>0.8599 +/- 0.0204</td><td>0.9133 +/- 0.0139</td>
</tr>
</table>

*All values reported as mean +/- standard deviation across 3 folds (5,030 test samples).* 

**Key finding:** VGG3D outperformed ResNet3D across all metrics with substantially lower variance and faster training time (194.1 min vs 307.1 min), indicating better stability for this liver task.

#### ROC Curves

<p align="center">
<img src="results/liver/plots/comparison_roc.png" width="600" alt="Comparison ROC curves for liver tumor classification">
</p>

#### Training Curves

<p align="center">
<img src="results/liver/plots/resnet3d/training_curves_kfold.png" width="48%" alt="ResNet3D liver training curves">
<img src="results/liver/plots/vgg3d/training_curves_kfold.png" width="48%" alt="VGG3D liver training curves">
</p>

#### Grad-CAM Visualizations

**VGG3D:**
<p align="center">
<img src="results/liver/plots/gradcam/vgg3d/tp_1.png" width="24%">
<img src="results/liver/plots/gradcam/vgg3d/tn_1.png" width="24%">
<img src="results/liver/plots/gradcam/vgg3d/fp_1.png" width="24%">
<img src="results/liver/plots/gradcam/vgg3d/fn_1.png" width="24%">
</p>
<p align="center"><i>Left to right: True Positive, True Negative, False Positive, False Negative</i></p>

**ResNet3D:**
<p align="center">
<img src="results/liver/plots/gradcam/resnet3d/tp_1.png" width="24%">
<img src="results/liver/plots/gradcam/resnet3d/tn_1.png" width="24%">
<img src="results/liver/plots/gradcam/resnet3d/fp_1.png" width="24%">
<img src="results/liver/plots/gradcam/resnet3d/fn_1.png" width="24%">
</p>
<p align="center"><i>Left to right: True Positive, True Negative, False Positive, False Negative</i></p>

---

### Pancreas Cancer Detection — Complete

**Dataset:** [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/) — Pancreatic Cancer CT collection (80 labeled cases)

**Preprocessing:** CT DICOM series and NIfTI labels were paired by case ID, aligned to CT geometry, resampled to 1 mm isotropic, HU-windowed to [-150, 250], and converted into 96x96x96 patches. For this TCIA label set, positive voxels are encoded as label `1`, so tumor-positive extraction is forced accordingly.

<table>
<tr>
<th>Model</th><th>AUC-ROC</th><th>Sensitivity</th><th>Specificity</th><th>F1 Score</th><th>Accuracy</th>
</tr>
<tr>
<td><b>ResNet3D</b></td><td>0.9965 +/- 0.0024</td><td>0.9797 +/- 0.0152</td><td>0.9624 +/- 0.0235</td><td>0.9065 +/- 0.0489</td><td>0.9653 +/- 0.0201</td>
</tr>
<tr>
<td><b>VGG3D</b></td><td>0.9980 +/- 0.0010</td><td>0.9633 +/- 0.0198</td><td>0.9868 +/- 0.0063</td><td>0.9496 +/- 0.0149</td><td>0.9829 +/- 0.0051</td>
</tr>
</table>

*All values reported as mean +/- standard deviation across 3 folds (1,464 patches: 244 positive, 1,220 negative).* 

**Key finding:** Both models exceeded 0.99 mean AUC on pancreas classification. VGG3D achieved the strongest overall balance (higher specificity, F1, and accuracy), while ResNet3D reached the highest sensitivity.

#### ROC Curves

<p align="center">
<img src="results/pancreas/plots/comparison_roc.png" width="600" alt="Comparison ROC curves for pancreas tumor classification">
</p>

#### Training Curves

<p align="center">
<img src="results/pancreas/plots/resnet3d/training_curves_kfold.png" width="48%" alt="ResNet3D pancreas training curves">
<img src="results/pancreas/plots/vgg3d/training_curves_kfold.png" width="48%" alt="VGG3D pancreas training curves">
</p>

#### Grad-CAM Visualizations

**VGG3D:**
<p align="center">
<img src="results/pancreas/plots/gradcam/vgg3d/tp_1.png" width="24%">
<img src="results/pancreas/plots/gradcam/vgg3d/tn_1.png" width="24%">
<img src="results/pancreas/plots/gradcam/vgg3d/fp_1.png" width="24%">
<img src="results/pancreas/plots/gradcam/vgg3d/fn_1.png" width="24%">
</p>
<p align="center"><i>Left to right: True Positive, True Negative, False Positive, False Negative</i></p>

**ResNet3D:**
<p align="center">
<img src="results/pancreas/plots/gradcam/resnet3d/tp_1.png" width="24%">
<img src="results/pancreas/plots/gradcam/resnet3d/tn_1.png" width="24%">
<img src="results/pancreas/plots/gradcam/resnet3d/fp_1.png" width="24%">
<img src="results/pancreas/plots/gradcam/resnet3d/fn_1.png" width="24%">
</p>
<p align="center"><i>Left to right: True Positive, True Negative, False Positive, False Negative</i></p>

---

## Project Structure

```
.
├── src/
│   ├── architecture.py          # ResNet3D and VGG3D model definitions
│   ├── training.py              # Trainer with focal loss, AMP, gradient accumulation
│   ├── evaluator.py             # Metrics calculator (AUC, sensitivity, specificity, F1)
│   ├── fast_dataset.py          # Dataset class with 3D augmentation pipeline
│   ├── gradcam.py               # 3D Grad-CAM visualization engine
│   ├── utils.py                 # Seed management, checkpoint utilities
│   ├── preextract.py            # Lung: LUNA16 patch extraction pipeline
│   ├── main.py                  # Lung: training and evaluation entry point
│   ├── preextract_liver.py      # Liver: LiTS patch extraction pipeline
│   ├── main_liver.py            # Liver: training and evaluation entry point
│   ├── preextract_pancreas.py   # Pancreas: TCIA patch extraction pipeline
│   ├── verify_pancreas_patches.py # Pancreas: extraction integrity checks
│   └── main_pancreas.py         # Pancreas: training and evaluation entry point
├── results/
│   ├── lung/
│   │   ├── plots/               # Lung ROC curves, training curves, Grad-CAM figures
│   │   └── checkpoints/         # Trained model weights (not tracked in git)
│   ├── liver/
│   │   ├── plots/               # Liver ROC curves, training curves, Grad-CAM figures
│   │   └── checkpoints/         # Trained model weights (not tracked in git)
│   └── pancreas/
│       ├── plots/               # Pancreas ROC curves, training curves, Grad-CAM figures
│       └── checkpoints/         # Trained model weights (not tracked in git)
├── data/
│   └── LUNA16_csv_backup/       # Annotation and candidate CSVs
├── Proof/                       # Training session documentation
├── requirements.txt
└── README.md
```

## Usage

Each organ follows a two-step pipeline: **extract patches** from raw CT volumes, then **train and evaluate** both models with cross-validation.

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (tested on RTX 4050, 6 GB VRAM)
- ~30 GB free disk space per organ (for raw data + extracted patches; deletable after training)

### Installation

```bash
git clone https://github.com/LuxxyJr/Cancer-Detection-using-Deep-Learning.git
cd Cancer-Detection-using-Deep-Learning
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy scikit-learn matplotlib SimpleITK nibabel
```

### Running the Lung Pipeline

```bash
# Step 1: Download LUNA16 dataset to data/LUNA16/
# Step 2: Extract patches (run once, ~30 minutes)
python src/preextract.py

# Step 3: Train both models with 3-fold CV (~10 hours)
python src/main.py
```

### Running the Liver Pipeline

```bash
# Step 1: Download LiTS training CT volumes + tumor masks
# Step 2: Place them in data/Task03_Liver/imagesTr and data/Task03_Liver/labelsTr
# Step 3: Extract patches (run once, ~30-60 minutes)
python src/preextract_liver.py

# Step 4: Train both models with 3-fold CV
python src/main_liver.py
```

### Running the Pancreas Pipeline

```bash
# Step 1: Download TCIA pancreas CT + labels
# Step 2: Place raw data under data/Pancreas Dataset/
#         (manifest-*/Pancreas-CT/PANCREAS_XXXX/.../*.dcm and
#          TCIA_pancreas_labels-*/labelXXXX.nii.gz)
# Step 3: Extract patches (run once)
python src/preextract_pancreas.py

# Step 4: Verify extraction integrity
python src/verify_pancreas_patches.py

# Step 5: Train both models with 3-fold CV
python src/main_pancreas.py
```

After training completes for each organ, raw data and extracted patches can be safely deleted. Only checkpoints (~50 MB) and result plots are retained.

---

## Hardware Requirements

This project was designed and tested under strict hardware constraints to demonstrate that clinically relevant cancer detection is achievable without enterprise-grade infrastructure.

| Component | Specification |
|-----------|--------------|
| GPU (Lung experiments) | NVIDIA RTX 4050 Laptop (6 GB VRAM) |
| GPU (Liver experiments) | NVIDIA A100-SXM4-40GB MIG 3g.20gb (20 GB VRAM, AI Kosh cloud) |
| GPU (Pancreas experiments) | NVIDIA A100-SXM4-40GB MIG 3g.20gb (20 GB VRAM, AI Kosh cloud) |
| RAM | 16 GB DDR5 |
| Storage | NVMe SSD (30+ GB free per organ) |
| OS | Windows 11 |

Mixed-precision training and gradient accumulation enable an effective batch size of 16. Lung experiments demonstrate consumer-GPU feasibility; liver experiments demonstrate cloud-GPU scaling and improved throughput.

---

## License

This project is part of academic research. Please cite appropriately if used in derivative work.

---

## Author

**Sanchit Singh**
CGU Odisha | 2401020387@cgu-odisha.ac.in
