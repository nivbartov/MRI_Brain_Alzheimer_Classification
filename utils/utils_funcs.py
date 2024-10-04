import sys
import os
import numpy as np
import subprocess
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset, ConcatDataset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import matplotlib.pyplot as plt
import random
import time
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from prettytable import PrettyTable
import torchattacks
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image
from IPython.display import display
from sklearn.decomposition import PCA



def plot_random_images(dataset, num_imgs=20):
    # Sample random indices from the dataset
    random_indices = random.sample(range(len(dataset)), min(num_imgs, len(dataset)))
        
    # Calculate the number of rows needed (5 images per row)
    num_rows = (len(random_indices) + 4) // 5  # +4 to round up for any remainder
        
    # Set the figure size
    plt.figure(figsize=(15, 3 * num_rows))  # Adjust height based on number of rows
        
    # Define the label mapping with colors
    label_mapping = {
        0: ("Mild_Demented", "blue"),
        1: ("Moderate_Demented", "orange"),
        2: ("Non-Demented", "green"),
        3: ("Very_Mild_Demented", "red")
    }

    for i, idx in enumerate(random_indices):
        image, label = dataset[idx]
        image = image.permute(1, 2, 0)  # Change dimension order if needed
        plt.subplot(num_rows, 5, i + 1)  # 5 columns
        plt.imshow(image)
        
        # Get label text and color
        label_text, label_color = label_mapping.get(label, (str(label), "black"))  # No .item() since label is an int
        
        # Set title with color and label number
        plt.title(f"{label} - {label_text}", color=label_color)
        plt.axis('off')

    # Create a legend with colored labels at the top
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f"{i} - {label_mapping[i][0]}",
                        markerfacecolor=label_mapping[i][1], markersize=10) for i in range(4)]

    # Adjust legend position to be above all images
    plt.legend(handles=handles, title="Labels", bbox_to_anchor=(0.2, 5), loc='center right', ncol=4, 
           title_fontproperties={'weight': 'bold'})

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust space to fit legend above the images
    plt.show()


def save_model(model, optimizer, epoch, epoch_losses, model_name, cur_loss):
    """Save the model checkpoint."""
    print('==> Saving model ...')
    state = {
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,                   
        'epoch_losses': epoch_losses                   
    }
    
    current_time = datetime.now().strftime("%H%M%S_%d%m%Y")  # Format: HHMMSS_DDMMYYYY
    filename = f'./checkpoints/{model_name}_{current_time}_loss_{cur_loss}.pth'
    
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
        
    torch.save(state, filename)
    print(f'saved as {filename}')


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


def log_epoch(epoch, running_loss, train_accuracy, epoch_time):
    """Logs the statistics for the current epoch."""
    log = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% | ".format(
        epoch, running_loss, train_accuracy
    )
    log += "Epoch Time: {:.2f} secs".format(epoch_time)
    print(log)

def train(model, num_epochs, trainloader, device, criterion, optimizer, scheduler, kornia_aug=None, use_amp=False):
    epoch_losses = []
    model_name = type(model).__name__
    print(f"Training model: {model_name} on {device}")
    
    # GradScaler for mixed precision training
    scaler = GradScaler() if use_amp else None

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_time = time.time()
        
        # Use tqdm for the training progress bar
        for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f'Epoch {epoch}/{num_epochs}'):
            # Get the inputs
            inputs, labels = data
            # Send them to the device
            if kornia_aug is None:
                inputs = inputs.to(device)
            else:
                inputs = kornia_aug(inputs).to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision training
            with autocast(enabled=use_amp):
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Calculate the loss
            
            # Backpropagation and optimization with scaling if AMP is enabled
            if use_amp:
                scaler.scale(loss).backward()  # Backpropagation
                scaler.step(optimizer)  # Update parameters
                scaler.update()  # Update the scale for next iteration
            else:
                loss.backward()  # Backpropagation
                optimizer.step()  # Update parameters

            # Accumulate the loss
            running_loss += loss.item()

        # Normalizing the loss by the total number of train batches
        running_loss /= len(trainloader)
        epoch_losses.append(running_loss)

        # Calculate training set accuracy of the existing model
        train_accuracy = calculate_accuracy(model, trainloader, device)

        # Log the epoch statistics
        epoch_time = time.time() - epoch_time
        log_epoch(epoch, running_loss, train_accuracy, epoch_time)
        
        # Save model every 10 epochs
        if epoch % 5 == 0:
            save_model(model, optimizer, epoch, epoch_losses, model_name, running_loss)
        
        # Call scheduler step after each epoch
        scheduler.step()    

    return epoch_losses


def open_nvitop():
    try:
        # Open a new cmd window and run 'nvitop -m' command for GPU monitoring
        subprocess.run('start cmd /K "nvitop -m"', shell=True)
    
    except FileNotFoundError:
        print("nvitop not found. Make sure you have installed nvitop.")

def load_and_display_image(model_name, asset_name):
    # Construct the path to the image
    image_path = f"assets/{model_name}/{asset_name}.png"
    
    # Load the image using PIL
    img = Image.open(image_path)
    
    # Display the image inline in the notebook
    display(img)
    

def plot_loss_curve(epoch_losses, num_epochs):
    _, ax = plt.subplots(figsize=(5, 5))
    ax.plot(range(num_epochs), epoch_losses, color='red')
    ax.set_title('Train Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.grid(True)
    
    plt.show()
    
def extract_features(model, dataloader, device):
    """Extract features from DINO V2 model before the fully connected layers."""
    model.eval()  # Set model to evaluation mode
    features = []
    labels = []

    with torch.no_grad():
        for inputs, lbl in dataloader:
            inputs = inputs.to(device)  # Move inputs to the specified device (GPU)
            # Extract features from the model
            output = model(inputs)
            features.append(output)  # Store the tensor output
            labels.append(lbl)  # Store the tensor labels

    # Convert lists of tensors to NumPy arrays before returning
    features = np.concatenate([f.cpu().numpy() for f in features])  # Combine all feature arrays
    labels = np.concatenate([l.cpu().numpy() for l in labels])  # Combine all label arrays
    return features, labels

def plot_tsne(features, labels, n_components=2, perplexity=30, learning_rate=200, n_iter=1000):
    """Plot t-SNE visualization of the features with adjustable parameters."""
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Alzheimer Severity Class')
    plt.title(f't-SNE Visualization (Perplexity: {perplexity}, Learning Rate: {learning_rate}, Iterations: {n_iter})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid()
    plt.show()

def visualize_dino_tsne(model, dataloader, device, perplexity=30, learning_rate=200, n_iter=1000):
    """Main function to visualize t-SNE of DINO V2 output."""
    features, labels = extract_features(model, dataloader, device)
    plot_tsne(features, labels, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)

    
def apply_transformations(dataset, transform):
    processed_data = []
    for img, label in dataset:
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)  # Convert tensor to PIL Image
        img = transform(img)  # Apply transformation to the image
        processed_data.append((img, label))
    return processed_data

def print_accuracy_table(epsilons, accuracies):
    # Create a PrettyTable object
    table = PrettyTable()

    # Add columns
    table.field_names = ["Epsilon", "Accuracy"]
    for eps, acc in zip(epsilons, accuracies):
        table.add_row([eps, f"{acc:.4f}%"])

    # Print the table
    print(table)
    

def load_model(model, optimizer, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['net'])  
    optimizer.load_state_dict(checkpoint['optimizer']) 
    epoch = checkpoint['epoch'] 
    loss = checkpoint['loss']
    
    print(f"Loaded model from {model_path} (epoch: {epoch}, loss: {loss:.4f})")
    return epoch, loss


def plot_normalized_confusion_matrix(testloader, model, class_names, device, model_name):
    model.eval()  

    all_preds = []
    all_labels = []

    with torch.no_grad():  
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)  
            _, preds = torch.max(outputs, 1)  
            
            all_preds.extend(preds.cpu().numpy()) 
            all_labels.extend(labels.cpu().numpy())

    # Compute the normalized confusion matrix
    cm = confusion_matrix(all_labels, all_preds, normalize='true')

    # Create the directory to save the confusion matrix if it doesn't exist
    save_dir = os.path.join('assets', f'{model_name}')
    os.makedirs(save_dir, exist_ok=True)

    # Plot the normalized confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format=".2f")

    # Adjust title and labels
    plt.title(f'Normalized Confusion Matrix of {model_name} model', fontsize=16, pad=20)
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)

    # Spread out the class names
    plt.xticks(rotation=45, ha='right')  # Rotate x labels to avoid overlap
    plt.yticks(rotation=0)  # Keep y labels horizontal

    plt.tight_layout()  # Ensure everything fits without overlap

    # Save the plot to the specified directory
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.show()  # Display the confusion matrix in the notebook
    plt.close()  # Close the plot to free up memory



def test_single_point_attack(model, device, testloader, attack_type, epsilon, alpha=None, num_iter=None):
    # Accuracy counter
    correct = 0
    total = 0
    adv_examples = []

    # Set the model to evaluation mode
    model.eval()

    # Loop over all examples in the test set with batch size of 1
    for image, target in testloader:
        # Send the image and labels to the device
        image, target = image.to(device), target.to(device)

        # Ensure we're working with a single point (batch size should be 1)
        assert image.size(0) == 1, "The batch size should be 1 for this function."

        # No need to manually calculate gradients for attacks as torchattacks handles it
        if attack_type.lower() == 'fgsm':
            # Initialize FGSM attack from torchattacks
            attack = torchattacks.FGSM(model, eps=epsilon)
        elif attack_type.lower() == 'pgd':
            if alpha is None or num_iter is None:
                raise ValueError("Alpha and num_iter must be specified for PGD attack.")
            # Initialize PGD attack from torchattacks
            attack = torchattacks.PGD(model, eps=epsilon, alpha=alpha, steps=num_iter)
        else:
            raise ValueError("Unsupported attack type. Use 'fgsm' or 'pgd'.")

        # Generate the adversarial image
        perturbed_image = attack(image, target)

        # Re-classify the perturbed image (no need to track gradients here)
        with torch.no_grad():
            output = model(perturbed_image)

        # Get initial prediction and final prediction on the perturbed image
        init_pred = model(image).max(1, keepdim=True)[1]  # Initial prediction
        final_pred = output.max(1, keepdim=True)[1]  # Final prediction

        # Update the correct count
        correct += (final_pred == target).sum().item()
        total += target.size(0)

        # Save some adversarial examples for visualization
        if len(adv_examples) < 5:
            adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = 100. * correct / float(total)

    # Return the accuracy and adversarial examples
    return final_acc, adv_examples



def plot_adversarial_examples(epsilons, examples, figsize=(12, 15)):
    cnt = 0
    plt.figure(figsize=figsize)  # Set the figure size

    for i in range(1, len(epsilons)):  # Start from the second epsilon
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons) - 1, len(examples[0]), cnt)  # Adjust for i starting from 1
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
                
            orig, adv, ex = examples[i][j]  # Unpack the original and adversarial examples
            
            # Convert the tensor to a NumPy array, ensuring it's on the CPU
            ex_np = ex.transpose(1, 2, 0)  # Transpose if needed
            
            plt.title(f"{orig} -> {adv}")
            plt.imshow(ex_np)  # Display the adversarial example

    # Adjust spacing to prevent overlap between subplots
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(hspace=0.4, wspace=0.2)  # Increase space between subplots
    plt.show()


def adversarial_train(model, num_epochs, trainloader, device, criterion, optimizer, attack_type='fgsm',
                      epsilon=0.03, adv_weight=0.5, alpha=0.01, num_iter=10, kornia_aug=None):
    epoch_losses = []
    
    model_name = type(model).__name__
    print(f"Adversarial Training model: {model_name} on {device} with {attack_type.upper()} attack and adversarial weight {adv_weight}")

    # Initialize the attack based on the attack_type
    if attack_type.lower() == 'fgsm':
        attack = torchattacks.FGSM(model, eps=epsilon)
    elif attack_type.lower() == 'pgd':
        attack = torchattacks.PGD(model, eps=epsilon, alpha=alpha, steps=num_iter)
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}. Choose 'fgsm' or 'pgd'.")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_time = time.time()
        
        for batch_idx, data in enumerate(trainloader, 0):
            # Get the inputs and labels
            inputs, labels = data
            
            # Apply Kornia augmentations if provided
            if kornia_aug is None:
                inputs = inputs.to(device)
            else:
                inputs = kornia_aug(inputs).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Clean forward pass
            clean_outputs = model(inputs)
            clean_loss = criterion(clean_outputs, labels)
            
            # Generate adversarial examples using the torchattacks package
            adv_inputs = attack(inputs, labels)

            # Forward pass on adversarial examples
            adv_outputs = model(adv_inputs)
            adv_loss = criterion(adv_outputs, labels)

            # Combine losses from clean and adversarial examples using adv_weight
            loss = (1 - adv_weight) * clean_loss + adv_weight * adv_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()

        # Normalize the loss
        running_loss /= len(trainloader)
        epoch_losses.append(running_loss)

        # Calculate training accuracy
        train_accuracy = calculate_accuracy(model, trainloader, device)

        log = f"Epoch: {epoch} | Loss: {running_loss:.4f} | Training Accuracy: {train_accuracy:.3f}% | "
        epoch_time = time.time() - epoch_time
        log += f"Epoch Time: {epoch_time:.2f} secs"
        print(log)
        
        # Save the model every 5 epochs
        if epoch % 5 == 0:
            print('==> Saving model ...')
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': running_loss
            }

            current_time = datetime.now().strftime("%H%M%S_%d%m%Y")  # Format: HHMMSS_DDMMYYYY
            filename = f'./checkpoints/adv_attk_{model_name}_{current_time}.pth'

            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')

            torch.save(state, filename)
            print(f'Saved as {filename}')

    return epoch_losses


# def adversarial_train(model, num_epochs, trainloader, device, criterion, optimizer, attack_type='fgsm',
#                       epsilon=0.03, adv_weight=0.5, alpha=0.01, num_iter=10, kornia_aug=None):
#     epoch_losses = []
    
#     model_name = type(model).__name__
#     print(f"Adversarial Training model: {model_name} on {device} with {attack_type.upper()} attack and adversarial weight {adv_weight}")

#     # Initialize the attack based on the attack_type
#     if attack_type.lower() == 'fgsm':
#         attack = torchattacks.FGSM(model, eps=epsilon)
#     elif attack_type.lower() == 'pgd':
#         attack = torchattacks.PGD(model, eps=epsilon, alpha=alpha, steps=num_iter)
#     else:
#         raise ValueError(f"Unsupported attack type: {attack_type}. Choose 'fgsm' or 'pgd'.")

#     for epoch in range(1, num_epochs + 1):
#         model.train()
#         running_loss = 0.0
#         epoch_time = time.time()
        
#         # Use tqdm for progress bar
#         with tqdm(total=len(trainloader), desc=f'Epoch {epoch}/{num_epochs}', unit='batch') as pbar:
#             for batch_idx, data in enumerate(trainloader, 0):
#                 # Get the inputs and labels
#                 inputs, labels = data
                
#                 # Apply Kornia augmentations if provided
#                 if kornia_aug is None:
#                     inputs = inputs.to(device)
#                 else:
#                     inputs = kornia_aug(inputs).to(device)
#                 labels = labels.to(device)

#                 optimizer.zero_grad()

#                 # Clean forward pass
#                 clean_outputs = model(inputs)
#                 clean_loss = criterion(clean_outputs, labels)
                
#                 # Generate adversarial examples using the torchattacks package
#                 adv_inputs = attack(inputs, labels)

#                 # Forward pass on adversarial examples
#                 adv_outputs = model(adv_inputs)
#                 adv_loss = criterion(adv_outputs, labels)

#                 # Combine losses from clean and adversarial examples using adv_weight
#                 loss = (1 - adv_weight) * clean_loss + adv_weight * adv_loss
#                 loss.backward()
#                 optimizer.step()

#                 running_loss += loss.data.item()
#                 pbar.update(1)  # Update progress bar

#             # Normalize the loss
#             running_loss /= len(trainloader)
#             epoch_losses.append(running_loss)

#             # Calculate training accuracy
#             train_accuracy = calculate_accuracy(model, trainloader, device)

#             # Log the epoch statistics
#             epoch_time = time.time() - epoch_time
#             log_epoch(epoch, running_loss, train_accuracy, epoch_time)
        
#             # Save the model every 5 epochs
#             if epoch % 5 == 0:
#                 print('==> Saving model ...')
#                 state = {
#                     'net': model.state_dict(),
#                     'optimizer': optimizer.state_dict(),
#                     'epoch': epoch,
#                     'loss': running_loss
#                 }

#                 current_time = datetime.now().strftime("%H%M%S_%d%m%Y")  # Format: HHMMSS_DDMMYYYY
#                 filename = f'./checkpoints/adv_attk_{model_name}_{current_time}.pth'

#                 if not os.path.isdir('checkpoints'):
#                     os.mkdir('checkpoints')

#                 torch.save(state, filename)
#                 print(f'Saved as {filename}')

#     return epoch_losses


def calculate_accuracy_attack(model, dataloader, device, attack_type, epsilon, alpha=None, num_iter=None):
    model.eval()  # Set model to evaluation mode
    total_correct = 0
    total_images = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Generate perturbed images using the specified attack
        if attack_type.lower() == 'fgsm':
            # Initialize FGSM attack from torchattacks
            attack = torchattacks.FGSM(model, eps=epsilon)
            perturbed_images = attack(inputs, labels)

        elif attack_type.lower() == 'pgd':
            if alpha is None or num_iter is None:
                raise ValueError("Alpha and num_iter must be specified for PGD attack.")
            # Initialize PGD attack from torchattacks
            attack = torchattacks.PGD(model, eps=epsilon, alpha=alpha, steps=num_iter)
            perturbed_images = attack(inputs, labels)
        else:
            raise ValueError("Unsupported attack type. Use 'fgsm' or 'pgd'.")

        with torch.no_grad():  # Now, disable gradient tracking for accuracy calculation
            # Get model predictions on the perturbed images
            outputs = model(perturbed_images)
            _, predicted = torch.max(outputs.data, 1)

            # Update total images and correct predictions
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    # Calculate model accuracy
    model_accuracy = total_correct / total_images * 100
    return model_accuracy