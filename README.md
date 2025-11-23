# Image Colorization Extension for Needle Framework

This project implements an end-to-end image colorization system built on top of the Needle deep learning framework as part of the CMU Deep Learning Systems course.

## Overview

The system transforms grayscale images into colored outputs using a differentiable neural architecture within Needle's computational environment. It uses an encoder-decoder architecture with skip connections to predict color channels in the Lab color space.

## Features

- **Color Space Conversion**: Differentiable RGB ↔ Lab color space operations
- **Encoder-Decoder Network**: U-Net-style architecture with skip connections for colorization
- **Combined Loss Function**: L1 loss + SSIM + simplified perceptual loss
- **CIFAR-10 Dataset**: Complete implementation with colorization wrapper
- **Training & Evaluation**: Full training pipeline with metrics (PSNR, SSIM, L1 error)

## Architecture

### Colorization Network

The `ColorizationNet` uses an encoder-decoder architecture:

**Encoder** (progressive downsampling):
- Conv(1→64) @ 32×32
- Conv(64→128, stride=2) @ 16×16
- Conv(128→256, stride=2) @ 8×8
- Conv(256→512, stride=2) @ 4×4

**Bottleneck**:
- Conv(512→512) @ 4×4

**Decoder** (progressive upsampling with skip connections):
- UpConv(512→256) + skip(256) @ 8×8
- UpConv(512→128) + skip(128) @ 16×16
- UpConv(256→64) + skip(64) @ 32×32
- Conv(128→2) @ 32×32

**Input**: Grayscale image (N, 1, H, W) in [0, 1]  
**Output**: AB color channels (N, 2, H, W) in Lab space

### Loss Function

Combined loss with three components:
1. **L1 Loss** (weight=1.0): Mean absolute error between predicted and target AB channels
2. **SSIM Loss** (weight=0.1): Structural similarity for preserving image structure
3. **Perceptual Loss** (weight=0.05): Simplified feature-based loss

## Installation & Setup

### Prerequisites

```bash
# Python 3.8+
# CMake for building C++ backend
# pybind11 for Python bindings
```

### Build the Backend

```bash
cd /Users/adityabhatnagar/dls-proj
make
```

This compiles the C++ ndarray backend and creates the shared library.

### Verify Installation

```bash
python3 -c "import sys; sys.path.append('./python'); import needle; print('Needle imported successfully!')"
```

## Dataset

The project uses the CIFAR-10 dataset, which is already included in the `data/` directory.

**Dataset Statistics**:
- Training: 50,000 images (32×32 RGB)
- Test: 10,000 images (32×32 RGB)
- Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

The `ColorizationDataset` wrapper converts RGB images to grayscale-color pairs for training.

## Training

### Basic Training

```bash
cd /Users/adityabhatnagar/dls-proj
python3 train_colorization.py
```

### Configuration

Edit `train_colorization.py` to adjust hyperparameters:

```python
class Config:
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.001
    l1_weight = 1.0
    ssim_weight = 0.1
    perceptual_weight = 0.05
```

### Training Output

The training script will:
- Load CIFAR-10 dataset and create colorization pairs
- Initialize the colorization model
- Train for specified number of epochs
- Save checkpoints every 5 epochs
- Print training and validation losses

**Expected output**:
```
================================================================================
Image Colorization Training
================================================================================
Batch size: 64
Epochs: 20
Learning rate: 0.001
================================================================================
Loading CIFAR-10 dataset...
Training samples: 50000
Test samples: 10000

Initializing colorization model...
Setting up loss function...
Setting up optimizer...

Starting training...
================================================================================

Epoch 1/20
--------------------------------------------------------------------------------
Epoch [1/20] Batch [10/782] Loss: 0.3245 Time: 15.23s
...
```

### Checkpoints

Checkpoints are saved in `./checkpoints/` directory:
- `colorization_epoch_5.pkl`
- `colorization_epoch_10.pkl`
- `colorization_epoch_15.pkl`
- `colorization_epoch_20.pkl`

## Evaluation & Demo

### Jupyter Notebook Demo

```bash
cd /Users/adityabhatnagar/dls-proj
jupyter notebook demo_colorization.ipynb
```

The demo notebook includes:
1. Loading trained model
2. Visualizing colorization results
3. Computing quantitative metrics (PSNR, SSIM, L1 error)
4. Comparing predicted vs ground truth colorizations

### Metrics

**PSNR (Peak Signal-to-Noise Ratio)**:
- Measures reconstruction quality
- Higher is better (typically 20-40 dB for colorization)

**SSIM (Structural Similarity Index)**:
- Measures perceptual similarity
- Range [0, 1], higher is better

**L1 Error**:
- Mean absolute error in RGB space
- Lower is better

## Project Structure

```
/Users/adityabhatnagar/dls-proj/
├── python/needle/              # Needle framework
│   ├── ops/
│   │   ├── ops_colorspace.py   # NEW: Color space conversions
│   │   ├── ops_mathematic.py
│   │   ├── ops_logarithmic.py
│   │   └── __init__.py
│   ├── nn/
│   │   ├── nn_colorization.py  # NEW: Colorization network
│   │   ├── nn_losses.py        # NEW: Loss functions
│   │   ├── nn_basic.py
│   │   ├── nn_conv.py
│   │   └── __init__.py
│   ├── data/
│   │   ├── datasets/
│   │   │   ├── cifar10_dataset.py      # COMPLETED
│   │   │   ├── colorization_dataset.py # NEW
│   │   │   └── __init__.py
│   │   └── data_basic.py
│   ├── autograd.py
│   └── optim.py
├── src/                        # C++ backend
│   └── ndarray_backend_cpu.cc
├── apps/
│   └── models.py              # ResNet9 for reference
├── data/
│   └── cifar-10-batches-py/   # CIFAR-10 dataset
├── checkpoints/               # Saved model checkpoints
├── build/                     # CMake build directory
├── CMakeLists.txt
├── Makefile
├── train_colorization.py      # Training script
├── demo_colorization.ipynb    # Demo notebook
└── README.md                  # This file
```

## Implementation Details

### Color Space Operations

**RGB to Lab Conversion**:
1. Apply gamma correction (inverse sRGB)
2. Transform to XYZ using D65 illuminant matrix
3. Normalize by D65 white point
4. Convert XYZ to Lab using standard formulas

**Lab to RGB Conversion**:
1. Convert Lab to XYZ
2. Denormalize by D65 white point
3. Transform XYZ to RGB
4. Apply gamma correction (sRGB)
5. Clip to [0, 1]

### Network Components

**ConvBlock**: Conv + ReLU (BatchNorm omitted for MVP simplicity)

**UpConvBlock**: Nearest-neighbor upsampling (2×) + Conv + ReLU

**Skip Connections**: Concatenate encoder features with decoder features at matching resolutions

### Training Strategy

1. Convert RGB images to grayscale and extract ground-truth AB channels
2. Feed grayscale (L channel) to network
3. Network predicts AB channels
4. Compute combined loss against ground-truth AB
5. Backpropagate and update weights

## Results

Expected performance after 20 epochs on CIFAR-10:

- **Training Loss**: ~0.15-0.25
- **Validation Loss**: ~0.20-0.30
- **PSNR**: 20-25 dB
- **SSIM**: 0.7-0.85

*Note: These are approximate values. Actual performance depends on hyperparameters and training duration.*

## Challenges & Solutions

### Challenge 1: Numerical Stability in Lab Space
**Solution**: Carefully normalize Lab values and use epsilon constants in conversions

### Challenge 2: Skip Connection Implementation
**Solution**: Use stack + reshape operations to concatenate feature maps along channel dimension

### Challenge 3: Upsampling Operation
**Solution**: Implement nearest-neighbor upsampling using stack + reshape (simplified for MVP)

### Challenge 4: Gradient Flow
**Solution**: Approximate gradients for color space conversions (full Jacobian in production version)

## Future Improvements

1. **Full Gradient Implementation**: Compute exact Jacobians for color space conversions
2. **Better Upsampling**: Implement bilinear or learned transposed convolutions
3. **Attention Mechanisms**: Add self-attention for better long-range dependencies
4. **Larger Datasets**: Train on Places365 or ImageNet for more diverse scenes
5. **Class-Conditional Colorization**: Use class labels to guide colorization
6. **Perceptual Loss**: Integrate pretrained VGG features properly

## Technical Notes

- **Color Space**: Lab space separates luminance (L) from chrominance (a, b)
- **Format**: All images in NCHW format (Needle Conv requirement)
- **Device**: CPU backend (CUDA backend can be enabled if available)
- **Differentiability**: All operations support automatic differentiation

## References

1. **Colorful Image Colorization** (Zhang et al., ECCV 2016)
2. **U-Net: Convolutional Networks for Biomedical Image Segmentation** (Ronneberger et al., 2015)
3. **Perceptual Losses for Real-Time Style Transfer** (Johnson et al., ECCV 2016)
4. **Image Quality Assessment: From Error Visibility to Structural Similarity** (Wang et al., 2004)

## Acknowledgments

This project was developed as part of the CMU Deep Learning Systems course. The Needle framework provides an educational environment for understanding deep learning system design and implementation.

## License

This project is for educational purposes as part of the CMU DLS course.

---

**Author**: Aditya Bhatnagar  
**Course**: Deep Learning Systems (CMU)  
**Date**: November 2025

