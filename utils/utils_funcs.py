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
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms import ToPILImage
from torchcam.methods import CAM
from collections import Counter

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

def calculate_statistics(dataset, set_name):
    # Assuming each item in the dataset is a tuple (image, label)
    labels = [label for _, label in dataset]
    total_images = len(labels)
    
    # Count occurrences of each class
    class_counts = Counter(labels)
    
    # Print statistics
    print(f"\n{set_name} Statistics:")
    print(f"Total images: {total_images}")
    for cls, count in class_counts.items():
        print(f"Class {cls}: {count} images ({(count / total_images) * 100:.2f}%)")


def create_training_session(model_name):
    """Create a training session directory for saving model checkpoints and logs."""
    # Create a directory path based on the model name and current date
    current_time = datetime.now().strftime("%H%M%S_%d%m%Y")
    session_dir = f"./checkpoints/{model_name}_{current_time}"
    
    # Create the directory
    os.makedirs(session_dir, exist_ok=True)

    print(f"Created training session directory: {session_dir}")
    return session_dir

def save_model(model, optimizer, epoch, train_epoch_losses, validation_epoch_losses, 
               train_epoch_accuracies, validation_epoch_accuracies,  # Add accuracy lists
               model_name, cur_train_loss, cur_validation_loss, session_dir):
    """Save the model checkpoint to the specified session directory."""
    print('==> Saving model ...')

    # Save the model state, optimizer state, losses, and accuracies for all epochs
    state = {
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,                   
        'train_epoch_losses': train_epoch_losses,
        'validation_epoch_losses': validation_epoch_losses,
        'train_epoch_accuracies': train_epoch_accuracies,  # Save train accuracies
        'validation_epoch_accuracies': validation_epoch_accuracies  # Save validation accuracies
    }
    
    # Create a filename based on the model name, current time, training loss, and validation loss
    current_time = datetime.now().strftime("%H%M%S_%d%m%Y")  # Format: HHMMSS_DDMMYYYY
    filename = f'{session_dir}/{model_name}_{current_time}_train_{cur_train_loss:.4f}_val_{cur_validation_loss:.4f}.pth'
    
    # Ensure the session directory exists
    os.makedirs(session_dir, exist_ok=True)
    
    # Save the state to a file
    torch.save(state, filename)
    print(f'Saved as {filename}')
    
# Function to calculate the mean and standard deviation of a dataset
def calculate_mean_std(dataset):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_pixels = 0

    for image, _ in dataset:
        mean += image.mean(dim=(1, 2))
        std += image.pow(2).mean(dim=(1, 2))
        num_pixels += 1

    mean /= num_pixels
    std = torch.sqrt(std / num_pixels - mean.pow(2))

    return mean, std

# Function to create a new dataset with normalized images and scaled to [0, 1]
def get_normalized_dataset(dataset, mean, std):
    normalized_images = []
    labels = []
    for image, label in dataset:
        # Normalize the image tensor
        normalized_image = (image - mean[:, None, None]) / std[:, None, None]
        
        # Scale the normalized image to range [0, 1]
        scaled_image = (normalized_image + 1) / 2
        
        normalized_images.append(scaled_image)
        labels.append(label)
        
    return torch.stack(normalized_images), torch.tensor(labels)


def prepare_datasets(train_set, validation_set, test_set):
    # Calculate the mean and std for the training set
    train_mean, train_std = calculate_mean_std(train_set)
    print(f"Training Set Mean: {train_mean}")
    print(f"Training Set Std: {train_std}")

    # Normalize and scale all datasets using the mean and std from the training set
    train_images, train_labels = get_normalized_dataset(train_set, train_mean, train_std)
    validation_images, validation_labels = get_normalized_dataset(validation_set, train_mean, train_std)
    test_images, test_labels = get_normalized_dataset(test_set, train_mean, train_std)

    # Create TensorDatasets
    train_set = TensorDataset(train_images, train_labels)
    validation_set = TensorDataset(validation_images, validation_labels)
    test_set = TensorDataset(test_images, test_labels)
    
    print(f"Normalized using Mean: {train_mean} and Std: {train_std}.")
    print("Rescaled to [0,1]")

    return train_set, validation_set, test_set    
    

def calculate_accuracy(model, dataloader, device):
    model.eval()
    model.zero_grad = True
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

def log_epoch(epoch, train_loss, train_accuracy, validation_loss, validation_accuracy, epoch_time):
    """Logs the statistics for the current epoch."""
    log = "Epoch: {} | Training Loss: {:.4f} | Training Accuracy: {:.3f}% | ".format(
        epoch, train_loss, train_accuracy
    )
    log += "Validation Loss: {:.4f} | Validation Accuracy: {:.3f}% | ".format(
        validation_loss, validation_accuracy
    )
    log += "Epoch Time: {:.2f} secs".format(epoch_time)
    print(log)

def train_epoch(model, trainloader, device, criterion, optimizer, kornia_aug=None, use_amp=False):
    model.train()
    train_loss = 0.0
    
    # GradScaler for mixed precision training
    scaler = GradScaler() if use_amp else None

    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader), desc='Training'):
        inputs, labels = data
        if kornia_aug is None:
            inputs = inputs.to(device)
        else:
            inputs = kornia_aug(inputs).to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision training
        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Backpropagation and optimization
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        train_loss += loss.item()

    train_loss /= len(trainloader)
    train_accuracy = calculate_accuracy(model, trainloader, device)

    return train_loss, train_accuracy

def validate_epoch(model, validationloader, device, criterion, kornia_aug=None, use_amp=False):
    model.eval()
    validation_loss = 0.0

    with torch.no_grad():
        for i, data in tqdm(enumerate(validationloader, 0), total=len(validationloader), desc='Validation'):
            inputs, labels = data
            if kornia_aug is None:
                inputs = inputs.to(device)
            else:
                inputs = kornia_aug(inputs).to(device)
            labels = labels.to(device)

            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            validation_loss += loss.item()

    validation_loss /= len(validationloader)
    validation_accuracy = calculate_accuracy(model, validationloader, device)

    return validation_loss, validation_accuracy

def train_model(model, num_epochs, trainloader, validationloader, device, criterion, optimizer, scheduler, kornia_aug=None, use_amp=False):
    epoch_train_losses = []
    epoch_validation_losses = []
    epoch_train_accuracies = []  # Store train accuracies
    epoch_validation_accuracies = []  # Store validation accuracies
    model_name = type(model).__name__

    # Create a training session directory
    session_dir = create_training_session(model_name)

    print(f"Training model: {model_name} on {device}")

    for epoch in range(1, num_epochs + 1):
        epoch_time = time.time()

        # Training phase
        train_loss, train_accuracy = train_epoch(model, trainloader, device, criterion, optimizer, kornia_aug, use_amp)
        epoch_train_losses.append(train_loss)
        epoch_train_accuracies.append(train_accuracy)  # Save train accuracy

        # Validation phase
        validation_loss, validation_accuracy = validate_epoch(model, validationloader, device, criterion, kornia_aug, use_amp)
        epoch_validation_losses.append(validation_loss)
        epoch_validation_accuracies.append(validation_accuracy)  # Save validation accuracy

        # Log the epoch statistics
        epoch_time = time.time() - epoch_time
        log_epoch(epoch, train_loss, train_accuracy, validation_loss, validation_accuracy, epoch_time)

        # Save model every 5 epochs
        if epoch % 5 == 0:
            save_model(
                model,
                optimizer,
                epoch,
                epoch_train_losses,
                epoch_validation_losses,
                epoch_train_accuracies,  # Pass train accuracies to save_model
                epoch_validation_accuracies,  # Pass validation accuracies to save_model
                model_name,
                train_loss,
                validation_loss,
                session_dir  # Pass the session directory to save_model
            )

        # Call scheduler step after each epoch
        scheduler.step()

    return epoch_train_losses, epoch_validation_losses, epoch_train_accuracies, epoch_validation_accuracies

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

def plot_loss_curve(epoch_train_losses, epoch_validation_losses, num_epochs, model_name):
    plt.figure(figsize=(5, 5))
    
    # Plot training loss
    plt.plot(range(1, num_epochs + 1), epoch_train_losses, color='purple', label='Training Loss')
    
    # Plot validation loss
    plt.plot(range(1, num_epochs + 1), epoch_validation_losses, color='green', label='Validation Loss')
    
    # Add titles and labels
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Adjust x-ticks to avoid overlap by showing every 5th epoch
    plt.xticks(range(1, num_epochs + 1, 5))
    plt.grid(True)

    # Add legend
    plt.legend(loc='upper right')
    
    # Save the plot before showing it
    save_dir = os.path.join('assets', f'{model_name}')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'plot_loss_curve.png'))

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_accuracy_curve(epoch_train_accuracies, epoch_validation_accuracies, num_epochs, model_name):
    plt.figure(figsize=(5, 5))
    
    # Plot training accuracy
    plt.plot(range(1, num_epochs + 1), epoch_train_accuracies, color='red', label='Training Accuracy')
    
    # Plot validation accuracy
    plt.plot(range(1, num_epochs + 1), epoch_validation_accuracies, color='blue', label='Validation Accuracy')
    
    # Add titles and labels
    plt.title('Training and Validation Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # Adjust x-ticks to avoid overlap by showing every 5th epoch
    plt.xticks(range(1, num_epochs + 1, 5))
    plt.grid(True)

    # Add legend
    plt.legend(loc='lower right')
    
    # Create the directory to save the confusion matrix if it doesn't exist
    save_dir = os.path.join('assets', f'{model_name}')
    os.makedirs(save_dir, exist_ok=True)
    
    plt.savefig(os.path.join(save_dir, 'plot_accuracy_curve.png'))

    # Show the plot
    plt.tight_layout()
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

def print_accuracy_table(epsilons, accuracies, parameter_type):
    # Create a PrettyTable object
    table = PrettyTable()

    # Add columns
    table.field_names = [parameter_type, "Accuracy"]
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
    plt.title(f'Normalized Confusion Matrix', fontsize=16, pad=20)
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

def plot_adversarial_examples(parameter, examples, attack_name, parameter_type, figsize=(12, 15)):
    cnt = 0
    plt.figure(figsize=figsize)  # Set the figure size
    
    # Add a header title for the attack name
    plt.suptitle(f"Adversarial Examples under {attack_name} Attack", fontsize=16, fontweight='bold', y=0.95)

    for i in range(1, len(parameter)):  # Start from the second epsilon
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(parameter) - 1, len(examples[0]), cnt)  # Adjust for i starting from 1
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"{parameter_type}: {parameter[i]}", fontsize=14)
                
            orig, adv, ex = examples[i][j]  # Unpack the original and adversarial examples
            
            # Convert the tensor to a NumPy array, ensuring it's on the CPU
            ex_np = ex.transpose(1, 2, 0)  # Transpose if needed
            
            plt.title(f"{orig} -> {adv}")
            plt.imshow(ex_np)  # Display the adversarial example

    # Adjust spacing to prevent overlap between subplots
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.9, hspace=0.4, wspace=0.2)  # Increase space between subplots
    plt.show()

def adversarial_train_epoch(model, trainloader, device, criterion, optimizer,
                            attack, adv_weight, kornia_aug=None, use_amp=False):
    model.train()
    train_loss = 0.0
    
    # GradScaler for mixed precision training
    scaler = GradScaler() if use_amp else None

    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader), desc='Training'):
        inputs, labels = data
        if kornia_aug is None:
            inputs = inputs.to(device)
        else:
            inputs = kornia_aug(inputs).to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision training
        with autocast(enabled=use_amp):
            # Clean forward pass
            clean_outputs = model(inputs)
            clean_loss = criterion(clean_outputs, labels)

            # Generate adversarial examples
            adv_inputs = attack(inputs, labels)

            # Forward pass on adversarial examples
            adv_outputs = model(adv_inputs)
            adv_loss = criterion(adv_outputs, labels)

            # Combine losses from clean and adversarial examples
            loss = (1 - adv_weight) * clean_loss + adv_weight * adv_loss

        # Backpropagation and optimization
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        train_loss += loss.item()

    train_loss /= len(trainloader)
    train_accuracy = calculate_accuracy(model, trainloader, device)

    return train_loss, train_accuracy

def adversarial_validation_epoch(model, validationloader, device, criterion, attack, kornia_aug=None, use_amp=False):
    model.eval()
    validation_loss = 0.0
    correct = 0
    total = 0


    for data in tqdm(validationloader, desc='Validating'):
        inputs, labels = data
        if kornia_aug is None:
            inputs = inputs.to(device)
        else:
            inputs = kornia_aug(inputs).to(device)
        labels = labels.to(device)

        # Enable gradient computation for adversarial example generation
        inputs.requires_grad = True
        
        with autocast(enabled=use_amp):
            # Generate adversarial examples
            adv_inputs = attack(inputs, labels)

            with torch.no_grad():
                # Forward pass on adversarial examples
                outputs = model(adv_inputs)
                loss = criterion(outputs, labels)

        validation_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Turn off requires_grad after adversarial attack generation
        inputs.requires_grad = False

    validation_loss /= len(validationloader)
    validation_accuracy = 100 * correct / total

    return validation_loss, validation_accuracy

def adversarial_train_model(model, num_epochs, trainloader, validationloader, device, criterion, optimizer, 
                            attack_type='fgsm', epsilon=0.03, adv_weight=0.5, alpha=0.01, num_iter=10, 
                            kornia_aug=None, use_amp=False, scheduler=None):
    
    epoch_train_losses = []
    epoch_validation_losses = []
    epoch_train_accuracies = []  # Store train accuracies
    epoch_validation_accuracies = []  # Store validation accuracies

    model_name = type(model).__name__ + '_atk'
    print(f"Adversarial Training model: {model_name} on {device} with {attack_type.upper()} attack and adversarial weight {adv_weight}")

    # Initialize the attack based on the attack_type
    if attack_type.lower() == 'fgsm':
        attack = torchattacks.FGSM(model, eps=epsilon)
    elif attack_type.lower() == 'pgd':
        attack = torchattacks.PGD(model, eps=epsilon, alpha=alpha, steps=num_iter)
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}. Choose 'fgsm' or 'pgd'.")

    # Create a session directory for saving the model
    session_dir = create_training_session(model_name)

    for epoch in range(1, num_epochs + 1):
        epoch_time = time.time()

        # Training phase
        train_loss, train_accuracy = adversarial_train_epoch(
            model, trainloader, device, criterion, optimizer, attack, adv_weight, kornia_aug, use_amp)
        epoch_train_losses.append(train_loss)
        epoch_train_accuracies.append(train_accuracy)  # Save train accuracy

        # Validation phase
        validation_loss, validation_accuracy = adversarial_validation_epoch(
            model, validationloader, device, criterion, attack, kornia_aug, use_amp)
        epoch_validation_losses.append(validation_loss)
        epoch_validation_accuracies.append(validation_accuracy)  # Save validation accuracy

        # Log the epoch statistics
        epoch_time = time.time() - epoch_time
        log_epoch(epoch, train_loss, train_accuracy, validation_loss, validation_accuracy, epoch_time)

        # Save model every 5 epochs
        if epoch % 5 == 0:
            save_model(
                model,
                optimizer,
                epoch,
                epoch_train_losses,
                epoch_validation_losses,
                epoch_train_accuracies,  # Pass train accuracies to save_model
                epoch_validation_accuracies,  # Pass validation accuracies to save_model
                model_name,
                train_loss,
                validation_loss,
                session_dir  # Pass the session directory to save_model
            )

        # Step the scheduler after each epoch if provided
        if scheduler:
            scheduler.step()

    return epoch_train_losses, epoch_validation_losses, epoch_train_accuracies, epoch_validation_accuracies

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