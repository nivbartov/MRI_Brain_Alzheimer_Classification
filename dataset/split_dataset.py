# Import packages
import torch
from torch.utils.data import Subset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random

# Clone dataset from GitHub
!git clone https://github.com/nivbartov/MRI_Brain_Alzheimer_Classification
%cd MRI_Brain_Alzheimer_Classification/

# Load the dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.ImageFolder("./dataset/raw_dataset", transform=transform)

# Split the dataset into train, test and validation sets
dataset_size = len(dataset)
dataset_indices = list(range(dataset_size))
random.shuffle(dataset_indices)
train_size = int(0.7 * dataset_size)
test_size = int(0.2 * dataset_size)

train_indices = dataset_indices[ : train_size]
test_indices = dataset_indices[train_size : train_size + test_size]
validation_indices = dataset_indices[train_size + test_size : ]

train_set = Subset(dataset, train_indices)
test_set = Subset(dataset, test_indices)
validation_set = Subset(dataset, validation_indices)

# Save the dataset variables
torch.save(train_set, 'train_set.pt')
torch.save(validation_set, 'validation_set.pt')
torch.save(test_set, 'test_set.pt')