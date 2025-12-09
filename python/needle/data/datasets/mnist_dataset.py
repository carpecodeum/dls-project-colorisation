from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        
        raise NotImplementedError()
        

    def __getitem__(self, index) -> object:
        
        raise NotImplementedError()
        

    def __len__(self) -> int:
        
        raise NotImplementedError()
        