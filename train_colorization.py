"""Training script for image colorization."""

import sys
sys.path.append('./python')

import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

# Configuration
class Config:
    # Dataset
    data_dir = "./data/cifar-10-batches-py"
    
    # Training
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.001
    
    # Loss weights
    l1_weight = 1.0
    ssim_weight = 0.1
    perceptual_weight = 0.05
    
    # Device
    device = ndl.cpu()
    dtype = "float32"
    
    # Checkpoints
    checkpoint_dir = "./checkpoints"
    save_every = 5  # Save checkpoint every N epochs
    
    # Logging
    log_every = 10  # Log every N batches


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


def train_epoch(model, train_loader, loss_fn, optimizer, config, epoch, steps_per_epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        gray_np, ab_target_np = batch
        
        # Convert to Tensors
        gray = ndl.Tensor(gray_np, device=config.device, dtype=config.dtype)
        ab_target = ndl.Tensor(ab_target_np, device=config.device, dtype=config.dtype)
        
        # Forward pass
        ab_pred = model.predict_ab(gray)
        
        # Compute loss
        loss = loss_fn(ab_pred, ab_target)
        
        # Backward pass
        optimizer.reset_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        total_loss += loss.numpy()
        num_batches += 1
        
        if (batch_idx + 1) % config.log_every == 0 or (batch_idx + 1) == steps_per_epoch:
            avg_loss = total_loss / num_batches
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1}/{config.num_epochs}] "
                  f"Batch [{batch_idx+1}/{steps_per_epoch}] "
                  f"Loss: {avg_loss:.4f} "
                  f"Time: {elapsed:.2f}s")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, test_loader, loss_fn, config, steps_per_epoch):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in test_loader:
        gray_np, ab_target_np = batch
        
        # Convert to Tensors
        gray = ndl.Tensor(gray_np, device=config.device, dtype=config.dtype)
        ab_target = ndl.Tensor(ab_target_np, device=config.device, dtype=config.dtype)
        
        # Forward pass
        ab_pred = model.predict_ab(gray)
        
        # Compute loss
        loss = loss_fn(ab_pred, ab_target)
        
        total_loss += loss.numpy()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
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
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
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

