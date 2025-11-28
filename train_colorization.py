"""Training script for image colorization."""

import sys
sys.path.append('./python')

import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os


def check_cuda_available():
    """
    Check if CUDA/GPU is available for Needle.
    Returns (device, device_name, details)
    """
    details = []
    
    # Check 1: Does ndl.cuda exist?
    if not hasattr(ndl, 'cuda'):
        details.append("✗ ndl.cuda() not available (CUDA backend not compiled)")
        return ndl.cpu(), "CPU", details
    
    details.append("✓ ndl.cuda() function exists")
    
    # Check 2: Can we create a CUDA device?
    try:
        cuda_device = ndl.cuda()
        details.append("✓ CUDA device created")
    except Exception as e:
        details.append(f"✗ Failed to create CUDA device: {e}")
        return ndl.cpu(), "CPU", details
    
    # Check 3: Can we allocate a tensor on GPU?
    try:
        test_tensor = ndl.Tensor([1.0, 2.0, 3.0], device=cuda_device)
        details.append("✓ Tensor allocation on GPU successful")
    except Exception as e:
        details.append(f"✗ Failed to allocate tensor on GPU: {e}")
        return ndl.cpu(), "CPU", details
    
    # Check 4: Can we run operations?
    try:
        result = test_tensor + test_tensor
        _ = result.numpy()
        details.append("✓ GPU computation successful")
    except Exception as e:
        details.append(f"✗ GPU computation failed: {e}")
        return ndl.cpu(), "CPU", details
    
    return cuda_device, "CUDA/GPU", details


def print_device_info():
    """Print device detection results."""
    device, device_name, details = check_cuda_available()
    
    print("=" * 60)
    print("DEVICE DETECTION")
    print("=" * 60)
    for detail in details:
        print(f"  {detail}")
    print("-" * 60)
    print(f"  Selected device: {device_name}")
    print("=" * 60)
    
    return device


# Configuration
class Config:
    # Dataset
    data_dir = "./data/cifar-10-batches-py"
    
    # Training - reduced batch size to avoid OOM
    batch_size = 16  # Reduced from 64 to prevent out of memory
    num_epochs = 20
    learning_rate = 0.0005  # Reduced from 0.001 for stability (prevents NaN)
    
    # Loss weights - SSIM disabled for stability
    l1_weight = 1.0
    ssim_weight = 0.0  # Disabled - causes NaN
    perceptual_weight = 0.0  # Disabled for simplicity
    
    # Device will be set in main() after detection
    device = None
    dtype = "float32"
    
    # Checkpoints
    checkpoint_dir = "./checkpoints"
    save_every = 5  # Save checkpoint every N epochs
    
    # Logging
    log_every = 50  # Log every N batches (increased since more batches now)


def create_dataloaders(config):
    """Create training and validation dataloaders."""
    print("Loading CIFAR-10 dataset...")
    
    # Load CIFAR-10 dataset
    train_dataset = ndl.data.CIFAR10Dataset(
        base_folder=config.data_dir,
        train=True
    )
    
    test_dataset = ndl.data.CIFAR10Dataset(
        base_folder=config.data_dir,
        train=False
    )
    
    # Wrap with colorization dataset
    train_color_dataset = ndl.data.ColorizationDataset(train_dataset, return_rgb=False)
    test_color_dataset = ndl.data.ColorizationDataset(test_dataset, return_rgb=False)
    
    # Create dataloaders
    train_loader = ndl.data.DataLoader(
        train_color_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    test_loader = ndl.data.DataLoader(
        test_color_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    print(f"Training samples: {len(train_color_dataset)}")
    print(f"Test samples: {len(test_color_dataset)}")
    
    train_steps = (len(train_color_dataset) + config.batch_size - 1) // config.batch_size
    test_steps = (len(test_color_dataset) + config.batch_size - 1) // config.batch_size
    
    return train_loader, test_loader, train_steps, test_steps


def get_loss_value(loss):
    """Safely extract scalar loss value from tensor."""
    loss_np = loss.numpy()
    if hasattr(loss_np, 'flatten'):
        return float(loss_np.flatten()[0])
    elif hasattr(loss_np, 'item'):
        return float(loss_np.item())
    else:
        return float(loss_np)


def train_epoch(model, train_loader, loss_fn, optimizer, config, epoch, steps_per_epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    nan_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        gray_np, ab_target_np = batch
        
        # Check input for NaN
        if np.isnan(gray_np).any() or np.isnan(ab_target_np).any():
            nan_batches += 1
            continue
        
        # Normalize ab targets to smaller range for stability
        # ab channels are in [-128, 127], normalize to [-1, 1]
        ab_target_np = ab_target_np / 128.0
        
        # Convert to Tensors
        gray = ndl.Tensor(gray_np, device=config.device, dtype=config.dtype)
        ab_target = ndl.Tensor(ab_target_np, device=config.device, dtype=config.dtype)
        
        # Forward pass
        ab_pred = model.predict_ab(gray)
        
        # Compute loss
        loss = loss_fn(ab_pred, ab_target)
        
        # Check for NaN loss and skip if detected
        loss_val = get_loss_value(loss)
        if np.isnan(loss_val) or np.isinf(loss_val):
            nan_batches += 1
            if nan_batches <= 5:  # Only print first few warnings
                print(f"  [WARNING] NaN/Inf loss at batch {batch_idx+1}, skipping...")
            continue
        
        # Backward pass
        optimizer.reset_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        total_loss += loss_val
        num_batches += 1
        
        if (batch_idx + 1) % config.log_every == 0 or (batch_idx + 1) == steps_per_epoch:
            avg_loss = total_loss / max(num_batches, 1)
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1}/{config.num_epochs}] "
                  f"Batch [{batch_idx+1}/{steps_per_epoch}] "
                  f"Loss: {avg_loss:.4f} "
                  f"Time: {elapsed:.2f}s")
    
    if nan_batches > 0:
        print(f"  [WARNING] {nan_batches} batches skipped due to NaN/Inf")
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def evaluate(model, test_loader, loss_fn, config, steps_per_epoch):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in test_loader:
        gray_np, ab_target_np = batch
        
        # Skip NaN inputs
        if np.isnan(gray_np).any() or np.isnan(ab_target_np).any():
            continue
        
        # Normalize ab targets to [-1, 1]
        ab_target_np = ab_target_np / 128.0
        
        # Convert to Tensors
        gray = ndl.Tensor(gray_np, device=config.device, dtype=config.dtype)
        ab_target = ndl.Tensor(ab_target_np, device=config.device, dtype=config.dtype)
        
        # Forward pass
        ab_pred = model.predict_ab(gray)
        
        # Compute loss
        loss = loss_fn(ab_pred, ab_target)
        
        loss_val = get_loss_value(loss)
        if not (np.isnan(loss_val) or np.isinf(loss_val)):
            total_loss += loss_val
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, config):
    """Save model checkpoint."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(config.checkpoint_dir, f"colorization_epoch_{epoch+1}.pkl")
    
    # Save model parameters
    import pickle
    checkpoint = {
        'epoch': epoch,
        'model_state': model.parameters(),
        'optimizer_state': optimizer,
        'loss': loss,
    }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved: {checkpoint_path}")


def main():
    """Main training loop."""
    config = Config()
    
    print("=" * 80)
    print("Image Colorization Training")
    print("=" * 80)
    
    # Detect and select device (GPU if available, else CPU)
    config.device = print_device_info()
    
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print("=" * 80)
    
    # Create dataloaders
    train_loader, test_loader, train_steps, test_steps = create_dataloaders(config)
    
    # Create model
    print("\nInitializing colorization model...")
    model = nn.ColorizationModel(device=config.device, dtype=config.dtype)
    
    # Create loss function
    print("Setting up loss function...")
    loss_fn = nn.CombinedColorizationLoss(
        l1_weight=config.l1_weight,
        ssim_weight=config.ssim_weight,
        perceptual_weight=config.perceptual_weight,
        device=config.device,
        dtype=config.dtype
    )
    
    # Create optimizer
    print("Setting up optimizer...")
    optimizer = ndl.optim.Adam(
        model.parameters(),
        lr=config.learning_rate
    )
    
    print("\nStarting training...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print("-" * 80)
        
        # Train
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, config, epoch, train_steps)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss = evaluate(model, test_loader, loss_fn, config, test_steps)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, config)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, config)
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

