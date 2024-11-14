import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        self.X, self.y = [], []
        
        # Define the batch files
        batch_files = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        
        # Load each batch file
        for batch_file in batch_files:
            with open(os.path.join(base_folder, batch_file), 'rb') as file:
                batch = pickle.load(file, encoding='bytes')
                images = batch[b'data']
                labels = batch[b'labels']
                
                # Reshape and normalize images
                images = images.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
                self.X.append(images)
                self.y.extend(labels)
        
        # Stack all batches
        self.X = np.concatenate(self.X)
        self.y = np.array(self.y)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        image, label = self.X[index], self.y[index]
        
        # Apply transforms if provided
        if self.transforms:
            for transform in self.transforms:
                image = transform(image)
        
        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION
