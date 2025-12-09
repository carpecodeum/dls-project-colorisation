import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset


class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 Dataset with augmentation support.
    
    Supports train/val/test splits:
    - train=True, split='train': Training set (45,000 samples)
    - train=True, split='val': Validation set (5,000 samples)
    - train=False: Test set (10,000 samples)
    """
    
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[float] = 0.5,
        transforms: Optional[List] = None,
        split: str = 'train',
        val_size: int = 5000,
        seed: int = 42
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        p - probability for random augmentations
        transforms - list of transform functions to apply
        split - 'train' or 'val' (only used when train=True)
        val_size - number of samples for validation set
        seed - random seed for reproducible train/val split
        """
        
        self.transforms = transforms
        self.p = p
        
        if train:
            # Load all 5 training batches
            data_list = []
            labels_list = []
            for i in range(1, 6):
                batch_file = os.path.join(base_folder, f'data_batch_{i}')
                with open(batch_file, 'rb') as f:
                    batch_dict = pickle.load(f, encoding='bytes')
                    data_list.append(batch_dict[b'data'])
                    labels_list.append(batch_dict[b'labels'])
            
            X_all = np.concatenate(data_list, axis=0)
            y_all = np.concatenate(labels_list, axis=0)
            
            # Create train/val split with fixed seed
            np.random.seed(seed)
            indices = np.random.permutation(len(X_all))
            
            if split == 'val':
                indices = indices[:val_size]
            else:  # train
                indices = indices[val_size:]
            
            self.X = X_all[indices]
            self.y = y_all[indices]
        else:
            # Load test batch
            test_file = os.path.join(base_folder, 'test_batch')
            with open(test_file, 'rb') as f:
                test_dict = pickle.load(f, encoding='bytes')
                self.X = test_dict[b'data']
                self.y = np.array(test_dict[b'labels'])
        
        # Normalize to [0, 1]
        self.X = self.X.astype(np.float32) / 255.0
        
        # Reshape from (N, 3072) to (N, 3, 32, 32)
        self.X = self.X.reshape(-1, 3, 32, 32)
        

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        
        img = self.X[index].copy()
        label = self.y[index]
        
        if self.transforms:
            for transform in self.transforms:
                img = transform(img)
        
        return img, label
        

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        
        return len(self.X)
        
