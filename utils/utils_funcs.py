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




def save_model(model, optimizer, epoch, running_loss, model_name):
    """Save the model checkpoint."""
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


def train(model, num_epochs, trainloader, device, criterion, optimizer, scheduler, kornia_aug=None):
    epoch_losses = []
    
    model_name = type(model).__name__
    print(f"Training model: {model_name} on {device}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_time = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # send them to device
            if kornia_aug is None:
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
        
        # save model every 5 epochs
        if epoch % 5 == 0:
            save_model(model, optimizer, epoch, running_loss, model_name)
        
        # Call scheduler step after each epoch
        scheduler.step()    

    return epoch_losses


def open_nvidia_smi():
    try:
        # Open a new cmd window and run the 'nvidia-smi -l 1' command to dynamically update GPU stats
        subprocess.run('start cmd /K "nvidia-smi -l 1"', shell=True)
    
    except FileNotFoundError:
        print("nvidia-smi not found. Make sure you have NVIDIA drivers installed.")

    

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









# def adversarial_train(model, num_epochs, trainloader, device, criterion, optimizer, attack_type='fgsm',
#                       epsilon=0.03,adv_weight=0.5, alpha=0.01, num_iter=10, kornia_aug=None):
#     epoch_losses = []
    
#     model_name = type(model).__name__
#     print(f"Adversarial Training model: {model_name} on {device} with {attack_type.upper()} attack and adversarial weight {adv_weight}")

#     for epoch in range(1, num_epochs + 1):
#         model.train()
#         running_loss = 0.0
#         epoch_time = time.time()
        
#         for batch_idx, data in enumerate(trainloader, 0):
#             # get the inputs and labels
#             inputs, labels = data
            
#             # Apply Kornia augmentations if provided
#             if kornia_aug is None:
#                 inputs = inputs.to(device)
#             else:
#                 inputs = kornia_aug(inputs).to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()

#             # Clean forward pass
#             clean_outputs = model(inputs)
#             clean_loss = criterion(clean_outputs, labels)
            
#             # Generate adversarial examples
#             if attack_type == 'fgsm':
#                 adv_inputs = fgsm_attack(model, inputs, labels, epsilon, criterion)
#             # elif attack_type == 'pgd':
#             #     adv_inputs = pgd_attack(model, inputs, labels, epsilon, alpha, num_iter)
            
#             # Forward pass on adversarial examples
#             adv_outputs = model(adv_inputs)
#             adv_loss = criterion(adv_outputs, labels)

#             # Combine losses from clean and adversarial examples using adv_weight
#             loss = (1 - adv_weight) * clean_loss + adv_weight * adv_loss
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.data.item()

#         # Normalize the loss
#         running_loss /= len(trainloader)
#         epoch_losses.append(running_loss)

#         # Calculate training accuracy
#         train_accuracy = calculate_accuracy(model, trainloader, device)

#         log = f"Epoch: {epoch} | Loss: {running_loss:.4f} | Training Accuracy: {train_accuracy:.3f}% | "
#         epoch_time = time.time() - epoch_time
#         log += f"Epoch Time: {epoch_time:.2f} secs"
#         print(log)
        
#         # Save the model every 5 epochs
#         if epoch % 5 == 0:
#             print('==> Saving model ...')
#             state = {
#                 'net': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'epoch': epoch,
#                 'loss': running_loss
#             }

#             current_time = datetime.now().strftime("%H%M%S_%d%m%Y")  # Format: HHMMSS_DDMMYYYY
#             filename = f'./checkpoints/adv_attk_{model_name}_{current_time}.pth'

#             if not os.path.isdir('checkpoints'):
#                 os.mkdir('checkpoints')

#             torch.save(state, filename)
#             print(f'Saved as {filename}')
    
#     return epoch_losses




# def fgsm_attack(model, inputs, labels, epsilon, criterion):
#     # Ensure the model is in evaluation mode
#     model.eval()
    
#     # Set requires_grad attribute of tensor to track gradients
#     inputs.requires_grad = True
    
#     # Perform the forward pass
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)
    
#     # Zero all existing gradients
#     model.zero_grad()
    
#     # Backward pass to calculate gradients of the loss with respect to inputs
#     loss.backward()
    
#     # Collect the sign of the gradients
#     grad_sign = inputs.grad.data.sign()
    
#     # Create the perturbed image by adjusting each pixel of the input image
#     perturbed_inputs = inputs + epsilon * grad_sign
    
#     # Clip the perturbation to ensure the pixel values are valid (i.e., between 0 and 1)
#     perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
    
#     return perturbed_inputs



# def calculate_accuracy_attack(model, dataloader, device, attack_type, epsilon, criterion, alpha=None, num_iter=None):
#     model.eval()  # Set model to evaluation mode
#     total_correct = 0
#     total_images = 0

#     for inputs, labels in dataloader:
        
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         # Generate perturbed images using the specified attack
#         if attack_type.lower() == 'fgsm':
#             # No torch.no_grad() here because gradients are needed for FGSM
#             perturbed_images = fgsm_attack(model, inputs, labels, epsilon, criterion)
#         # elif attack_type.lower() == 'pgd':
#         #     if alpha is None or num_iter is None:
#         #         raise ValueError("Alpha and num_iter must be specified for PGD attack.")
#         #     perturbed_images = pgd_attack(model, inputs, labels, epsilon, alpha, num_iter, criterion)
#         else:
#             raise ValueError("Unsupported attack type. Use 'fgsm' or 'pgd'.")

#         with torch.no_grad():  # Now, disable gradient tracking for accuracy calculation
#             # Get model predictions on the perturbed images
#             outputs = model(perturbed_images)
#             _, predicted = torch.max(outputs.data, 1)

#             # Update total images and correct predictions
#             total_images += labels.size(0)
#             total_correct += (predicted == labels).sum().item()

#     # Calculate model accuracy
#     model_accuracy = total_correct / total_images * 100
#     return model_accuracy




# def fgsm_attack_single_point(image, epsilon, data_grad):
#     # Collect the element-wise sign of the data gradient
#     sign_data_grad = data_grad.sign()
#     # Create the perturbed image by adjusting each pixel of the input image
#     perturbed_image = image + epsilon*sign_data_grad
#     # Adding clipping to maintain [0,1] range
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)
#     # Return the perturbed image
#     return perturbed_image 

# def denorm(batch, device, mean=[0.1307], std=[0.3081]):

#     if isinstance(mean, list):
#         mean = torch.tensor(mean).to(device)
#     if isinstance(std, list):
#         std = torch.tensor(std).to(device)

#     return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


# def test_fgsm_single_point_attack(model, device, testloader, epsilon, criterion):
#     # Accuracy counter
#     correct = 0
#     total = 0
#     adv_examples = []

#     # Set the model to evaluation mode
#     model.eval()

#     # Loop over all examples in the test set with batch size of 1
#     for data, target in testloader:
#         # Send the data and labels to the device
#         data, target = data.to(device), target.to(device)

#         # Ensure we're working with a single point
#         assert data.size(0) == 1, "The batch size should be 1 for this function."

#         # Set requires_grad attribute of tensor for the attack
#         data.requires_grad = True

#         # Forward pass
#         output = model(data)
#         init_pred = output.max(1, keepdim=True)[1]  # Get the index of the max log-probability

#         # Calculate the loss
#         loss = criterion(output, target)

#         # Zero all existing gradients
#         model.zero_grad()

#         # Backward pass to calculate gradients
#         loss.backward()

#         # Collect the gradients for this point
#         data_grad = data.grad.data

#         # Restore the data to its original scale (denormalization)
#         # data_denorm = denorm(data, device)
#         data_denorm = data
#         # Apply FGSM attack
#         perturbed_data = fgsm_attack_single_point(data_denorm, epsilon, data_grad)

#         # Reapply normalization to the perturbed data
#         perturbed_data_normalized = perturbed_data
#         # perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

#         # Disable gradients for inference (no need for gradients now)
#         with torch.no_grad():
#             # Re-classify the perturbed image
#             output = model(perturbed_data_normalized)

#         # Get final prediction
#         final_pred = output.max(1, keepdim=True)[1]

#         # Update the correct count
#         correct += (final_pred == target).sum().item()
#         total += target.size(0)

#         # Save some adversarial examples for visualization
#         if len(adv_examples) < 5:
#             adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
#             adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

#         # After processing each point, delete the gradients to save memory
#         # data.grad = None  # Explicitly clear gradients for this data point
#         # torch.cuda.empty_cache()  # Clear the CUDA cache to free up memory

#     # Calculate final accuracy for this epsilon
#     final_acc = correct / float(total)
#     # print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {total} = {final_acc}")
    
#     # Return the accuracy and adversarial examples
#     return final_acc, adv_examples