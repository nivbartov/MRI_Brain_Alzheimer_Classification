import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset, ConcatDataset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential
import matplotlib.pyplot as plt
import random
import time
from datetime import datetime
from sklearn.manifold import TSNE

def calculate_accuracy(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_images = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    model_accuracy = total_correct / total_images * 100
    return model_accuracy

def train(model, num_epochs, trainloader, device, criterion, optimizer, kornia_aug=None):
    epoch_losses = []
    
    model_name = type(model).__name__
    print(f"Training model: {model_name}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_time = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # send them to device
            if kornia_aug == None:
                inputs = inputs.to(device)
            else:
                inputs = kornia_aug(inputs).to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = model.forward(inputs) # forward pass
            loss = criterion(outputs, labels) # calculate the loss
            optimizer.zero_grad() # zero the parameter gradients
            loss.backward() # backpropagation
            optimizer.step() # update parameters

            # print statistics
            running_loss += loss.data.item()

        # Normalizing the loss by the total number of train batches
        running_loss /= len(trainloader)
        epoch_losses.append(running_loss)

        # Calculate training set accuracy of the existing model
        train_accuracy = calculate_accuracy(model, trainloader, device)

        log = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% | ".format(epoch, running_loss, train_accuracy)
        epoch_time = time.time() - epoch_time
        log += "Epoch Time: {:.2f} secs".format(epoch_time)
        print(log)
        
        # save model
        if epoch % 5 == 0:
            print('==> Saving model ...')
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,                      
                'loss': running_loss                   
            }
            
            current_time = datetime.now().strftime("%H%M%S_%d%m%Y")  # Format: HHMMSS_DDMMYYYY
            filename = f'./checkpoints/{model_name}_{current_time}.pth'
            
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
                
            torch.save(state, filename)
            print(f'saved as {filename}')
    
    return epoch_losses

def plot_loss_curve(epoch_losses, num_epochs):
    _, ax = plt.subplots(figsize=(5, 5))
    ax.plot(range(num_epochs), epoch_losses, color='red')
    ax.set_title('Train Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.grid(True)
    
    plt.show()
    
def plot_tsne(data_set, num_classes, header):
    
    # Check if data_set is a PyTorch Dataset or Subset
    if isinstance(data_set, torch.utils.data.Dataset):
        # Assuming the dataset is a TensorDataset or contains (features, labels)
        features = []
        labels = []
        
        # Iterate through the dataset to collect features and labels
        for data, label in data_set:
            features.append(data.view(-1))  # Flatten the features
            labels.append(label)
        
        # Convert lists to tensors
        features = torch.stack(features)
        labels = torch.tensor(labels)
    else:
        # If the data_set is a dictionary (if this is ever the case)
        features = data_set['features']
        labels = data_set['labels']

        # Flatten the features if needed (for example, if they are images)
        if len(features.shape) > 2:  # Assuming shape [batch_size, channels, width, height]
            features = features.view(features.size(0), -1)

    # Apply t-SNE to reduce features to 2D for visualization
    tsne = TSNE(n_components=2)
    transformed_features = tsne.fit_transform(features)

    # Plot the t-SNE result
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(transformed_features[:, 0], transformed_features[:, 1], c=labels, cmap=plt.cm.get_cmap('jet', num_classes))
    plt.colorbar(scatter)
    plt.title(f't-SNE Data Distribution for {header}')
    plt.show()

    
def apply_transformations(dataset, transform):
    processed_data = []
    for img, label in dataset:
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)  # Convert tensor to PIL Image
        img = transform(img)  # Apply transformation to the image
        processed_data.append((img, label))
    return processed_data


def load_model(model, optimizer, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['net'])  
    optimizer.load_state_dict(checkpoint['optimizer']) 
    epoch = checkpoint['epoch'] 
    loss = checkpoint['loss']
    
    print(f"Loaded model from {model_path} (epoch: {epoch}, loss: {loss:.4f})")
    return epoch, loss