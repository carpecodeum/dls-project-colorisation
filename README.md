# Image Colorization - Data Pipeline & Baseline Metrics

This module provides the data loading, augmentation, and baseline evaluation infrastructure for image colorization on CIFAR-10.

## Completed Deliverables

### 1. Dataloader with Augmentation Pipeline
- **Complete CIFAR-10 loader** with train/val/test split support
- **Augmentation pipeline** including:
  - Random horizontal flip (50%)
  - Random crop & resize (30%)
  - Color jitter - brightness/contrast (20%)
  - Random 90-degree rotation (20%)

### 2. Train/Val/Test Splits
- **Training**: 45,000 samples (with augmentation)
- **Validation**: 5,000 samples (no augmentation)
- **Test**: 10,000 samples (no augmentation)

### 3. Baseline Metrics Script
- Computes PSNR and SSIM on grayscale baseline
- Runs on configurable number of samples (default: 100)
- Outputs JSON results for tracking

### 4. Visualization Notebook
- Dataset split statistics
- Sample visualizations from each split
- Augmentation effects visualization
- Baseline metrics distribution plots

## Quick Start

### Get the CIFAR-10 data
```bash
cd <project folder>
!wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
!tar -xzf cifar-10-python.tar.gz
!mkdir -p data
!mv cifar-10-batches-py data/
```

### Verify Installation
```bash
python3 -c "
import sys; sys.path.append('./python')
import needle.data as data
train, val, test = data.create_colorization_splits('./data/cifar-10-batches-py')
print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')
"
```

### Run Baseline Metrics
```bash
python3 baseline_metrics.py --samples 100 --output baseline_results.json
```

### Open Visualization Notebook
```bash
jupyter notebook visualization.ipynb
```

## File Structure

```
dls-proj/
├── python/needle/data/datasets/
│   ├── cifar10_dataset.py      # Complete CIFAR-10 loader with splits
│   └── colorization_dataset.py # Colorization wrapper + augmentation
├── baseline_metrics.py         # PSNR/SSIM evaluation script
├── visualization.ipynb         # Data visualization notebook
├── baseline_results.json       # Metrics output (generated)
└── README.md                   # This file
```

## Usage Examples

### Create Dataset Splits
```python
import needle.data as data

train_ds, val_ds, test_ds = data.create_colorization_splits(
    base_folder='./data/cifar-10-batches-py',
    val_size=5000,
    seed=42,
    augment_train=True
)
```

### Get Sample Data
```python
# Returns (grayscale, ab_channels, rgb_original)
L, ab, rgb = train_ds[0]
# L: (1, 32, 32) - grayscale in [0, 1]
# ab: (2, 32, 32) - Lab color channels
# rgb: (3, 32, 32) - original RGB in [0, 1]
```

### Custom Augmentation
```python
from needle.data.datasets.colorization_dataset import ColorizationAugmentation

aug = ColorizationAugmentation(
    flip_prob=0.5,
    crop_prob=0.3,
    jitter_prob=0.2,
    rotation_prob=0.2
)

augmented_img = aug(img)
```

## Baseline Metrics Results (100 samples)

| Metric | Mean | Std |
|--------|------|-----|
| PSNR (dB) | 23.02 | 3.75 |
| SSIM (windowed) | 0.7993 | 0.1594 |
| L1 Error | 0.0581 | 0.0291 |

*Note: These are baseline values (grayscale only, no colorization). Trained models should improve these metrics.*

## Dependencies

- numpy
- matplotlib (for visualization)
- jupyter (for notebook)

### Dependency Installation
Inside the jupyter notebook, run the following command: 
```bash
!pip install -r requirements.txt
```
