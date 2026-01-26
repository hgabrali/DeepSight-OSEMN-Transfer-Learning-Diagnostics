import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import numpy as np

def get_cifar10_data(subset_ratio=0.2):
    """
    Obtain CIFAR-10 and perform Stratified Sampling for rapid prototyping.
    """
    # Download full dataset
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    
    # Get labels for stratification
    labels = np.array(full_train_dataset.targets)
    
    # Perform Stratified Split
    train_indices, _ = train_test_split(
        np.arange(len(labels)),
        test_size=1 - subset_ratio,
        stratify=labels,
        random_state=42
    )
    
    # Create the subset
    train_subset = Subset(full_train_dataset, train_indices)
    print(f"✅ Data Obtained: Subset size is {len(train_subset)} (Ratio: {subset_ratio})")
    
    return train_subset
