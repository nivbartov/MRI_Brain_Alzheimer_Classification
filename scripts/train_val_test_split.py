import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import numpy as np
import random

def split_dataset(image_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, verbose=False):
    """
    Split the dataset into training, validation, and test datasets.

    Args:
    - image_folder (datasets.ImageFolder): The dataset to split.
    - train_ratio (float): The proportion of the dataset to include in the training set.
    - val_ratio (float): The proportion of the dataset to include in the validation set.
    - test_ratio (float): The proportion of the dataset to include in the test set.
    - seed (int): Random seed for reproducibility.

    Returns:
    - train_dataset (Subset): Subset for the training set.
    - val_dataset (Subset): Subset for the validation set.
    - test_dataset (Subset): Subset for the test set.
    """

    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Calculate sizes for each dataset
    total_size = len(image_folder)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Shuffle and split the dataset
    train_dataset, val_dataset, test_dataset = random_split(image_folder, [train_size, val_size, test_size])

    # Print the sizes of each dataset
    if(verbose):
        print(f'Training set size: {len(train_dataset)}')
        print(f'Validation set size: {len(val_dataset)}')
        print(f'Test set size: {len(test_dataset)}')

    return train_dataset, val_dataset, test_dataset