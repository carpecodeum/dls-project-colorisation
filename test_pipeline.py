"""Quick test to verify the colorization pipeline works end-to-end."""

import sys
sys.path.append('./python')

import needle as ndl
import needle.nn as nn

print("=" * 80)
print("Testing Image Colorization Pipeline")
print("=" * 80)

# Test 1: Dataset loading
print("\n1. Testing dataset loading...")
try:
    train_dataset = ndl.data.CIFAR10Dataset('./data/cifar-10-batches-py', train=True)
    test_dataset = ndl.data.CIFAR10Dataset('./data/cifar-10-batches-py', train=False)
    print(f"   ✓ CIFAR-10 loaded: {len(train_dataset)} train, {len(test_dataset)} test")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Colorization dataset
print("\n2. Testing colorization dataset wrapper...")
try:
    color_dataset = ndl.data.ColorizationDataset(train_dataset, return_rgb=False)
    gray, ab = color_dataset[0]
    assert gray.shape == (1, 32, 32), f"Expected gray shape (1, 32, 32), got {gray.shape}"
    assert ab.shape == (2, 32, 32), f"Expected ab shape (2, 32, 32), got {ab.shape}"
    print(f"   ✓ Colorization dataset works: gray {gray.shape}, ab {ab.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Model initialization
print("\n3. Testing model initialization...")
try:
    device = ndl.cpu()
    dtype = "float32"
    model = nn.ColorizationModel(device=device, dtype=dtype)
    print(f"   ✓ Model created with {len(model.parameters())} parameter groups")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 4: Forward pass
print("\n4. Testing forward pass...")
try:
    # Reshape gray to batch format - gray is already a numpy array from dataset
    gray_reshaped = gray.reshape(1, 1, 32, 32)
    gray_tensor = ndl.Tensor(gray_reshaped, device=device, dtype=dtype)
    ab_pred = model.predict_ab(gray_tensor)
    assert ab_pred.shape == (1, 2, 32, 32), f"Expected output shape (1, 2, 32, 32), got {ab_pred.shape}"
    print(f"   ✓ Forward pass successful: output shape {ab_pred.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Loss functions
print("\n5. Testing loss functions...")
try:
    ab_reshaped = ab.reshape(1, 2, 32, 32)
    ab_target = ndl.Tensor(ab_reshaped, device=device, dtype=dtype)
    
    # L1 Loss
    l1_loss = nn.L1Loss()
    l1_val = l1_loss(ab_pred, ab_target)
    l1_scalar = float(l1_val.numpy().flatten()[0])
    print(f"   ✓ L1 Loss: {l1_scalar:.4f}")
    
    # SSIM Loss
    ssim_loss = nn.SSIMLoss()
    ssim_val = ssim_loss(ab_pred, ab_target)
    ssim_scalar = float(ssim_val.numpy().flatten()[0])
    print(f"   ✓ SSIM Loss: {ssim_scalar:.4f}")
    
    # Combined Loss
    combined_loss = nn.CombinedColorizationLoss(device=device, dtype=dtype)
    combined_val = combined_loss(ab_pred, ab_target)
    combined_scalar = float(combined_val.numpy().flatten()[0])
    print(f"   ✓ Combined Loss: {combined_scalar:.4f}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Backward pass
print("\n6. Testing backward pass...")
try:
    loss = combined_loss(ab_pred, ab_target)
    loss.backward()
    print(f"   ✓ Backward pass successful")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Optimizer
print("\n7. Testing optimizer...")
try:
    optimizer = ndl.optim.Adam(model.parameters(), lr=0.001)
    print(f"   ✓ Adam optimizer created")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 8: DataLoader
print("\n8. Testing dataloader...")
try:
    batch_size = 8
    dataloader = ndl.data.DataLoader(
        color_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    gray_batch_tensor, ab_batch_tensor = batch
    
    # Get numpy arrays for tensor creation
    gray_batch_np = gray_batch_tensor.numpy() if hasattr(gray_batch_tensor, 'numpy') else gray_batch_tensor
    ab_batch_np = ab_batch_tensor.numpy() if hasattr(ab_batch_tensor, 'numpy') else ab_batch_tensor
    
    # Convert to Tensors on target device
    gray_batch = ndl.Tensor(gray_batch_np, device=device, dtype=dtype)
    ab_batch = ndl.Tensor(ab_batch_np, device=device, dtype=dtype)
    
    print(f"   ✓ DataLoader works: batch size {batch_size}")
    print(f"     Gray batch shape: {gray_batch.shape}")
    print(f"     AB batch shape: {ab_batch.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: One training iteration
print("\n9. Testing one training iteration...")
try:
    # Forward
    ab_pred = model.predict_ab(gray_batch)
    
    # Loss
    loss = combined_loss(ab_pred, ab_batch)
    
    # Backward
    optimizer.reset_grad()
    loss.backward()
    
    # Update
    optimizer.step()
    
    loss_scalar = float(loss.numpy().flatten()[0])
    print(f"   ✓ Training iteration successful, loss: {loss_scalar:.4f}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("All tests passed! ✓")
print("The colorization pipeline is ready for training.")
print("=" * 80)
print("\nNext steps:")
print("  1. Run training: python3 train_colorization.py")
print("  2. View results: jupyter notebook demo_colorization.ipynb")
print("=" * 80)
