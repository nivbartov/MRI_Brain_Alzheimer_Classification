import sys
import os
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from functools import partial
import optuna
from torch import optim
import torchvision.transforms as transforms
import json
from kornia.augmentation import AugmentationSequential
from kornia import augmentation as K
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from models import def_models
from utils import utils_funcs

def save_best_params(best_params, model_name, loss_value, base_dir='checkpoints/op_tuna_params'):
    """Save the best parameters to a structured directory."""
    # Create the base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Create a model-specific directory
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Create the filename
    filename = f"{model_name}_{loss_value:.4f}.json"
    file_path = os.path.join(model_dir, filename)

    # Save the parameters to a JSON file
    with open(file_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print(f"Best parameters saved to {file_path}")


  
def define_model(trial, model_name, transfer_learning):
    models = ['DINOv2', 'ResNet', 'EfficientNet']
    
    if model_name not in models:
        raise ValueError(f"Model name {model_name} is not in the list of available models: {models}")
    
    output_channels = 4
    
    model_class = getattr(def_models, model_name)
    
    if model_name == 'DINOv2':
        # Load the DINOv2 backbone
        dino_v2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # Freeze the DINOv2 layers if using transfer learning
        if transfer_learning:
            for param in dino_v2_model.parameters():
                param.requires_grad = False
        
        # Create the DINOv2 model, passing the backbone
        model = model_class(DINOv2_backbone=dino_v2_model, output_channels=output_channels)
    elif model_name == 'ResNet':
        # Load the ResNet backbone
        resnet_model = torchvision.models.resnet34(pretrained=True)
        
        # Freeze the ResNet layers if using transfer learning
        if transfer_learning:
            for param in resnet_model.parameters():
                param.requires_grad = True
        
        # Create the ResNet model, passing the backbone
        model = model_class(ResNet_backbone=resnet_model, output_channels=output_channels)
    elif model_name == 'EfficientNet':
        # Load the EfficientNet backbone
        efficientnet_model = torchvision.models.efficientnet_b0(pretrained=True)
        
        # Freeze the EfficientNet layers if using transfer learning
        if transfer_learning:
            for param in efficientnet_model.parameters():
                param.requires_grad = True

        # Create the EfficientNet model, passing the backbone
        model = model_class(EfficientNet_backbone=efficientnet_model, output_channels=output_channels)
    else:
        # Handle other model types (NaiveModel, NaiveModelAug) as needed
        model = model_class(input_channels=3, output_channels=output_channels)

    return model


def objective(trial, model_name, epochs, device, loss_criterion, transfer_learning=True):
    # Generate the model
    model = define_model(trial, model_name, transfer_learning).to(device)
    
    # Hyperparameters to experiment: learning rate, optimizer, batch size
    # learning rate
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)  # log=True, will use log scale to interpolate between lr
    # optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    # batch size
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    # scheduler
    scheduler_name = trial.suggest_categorical('scheduler', ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"])
    
    if scheduler_name == "StepLR":
        scheduler = StepLR(optimizer, 10, 0.1)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, 30)
    else:  # ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    # Get MRI dataset - load the data and shuffle it
    train_set = torch.load('dataset/dataset_variables/train_set.pt')
    validation_set = torch.load('dataset/dataset_variables/validation_set.pt')
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3)
    validloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=3)
    
    # Limit train examples
    n_train_examples = 25 * batch_size
    n_valid_examples = 10 * batch_size
    
    augmentations = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.1),
        K.RandomVerticalFlip(p=0.1),
        K.RandomRotation(degrees=10, p=0.1),
        K.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), p=0.1),
        K.RandomBrightness(brightness=(0.8, 1.2), p=0.1),
        K.RandomContrast(contrast=(0.8, 1.2), p=0.1),
        K.RandomGamma(gamma=(0.9, 1.1), p=0.1),
        K.RandomSharpness(p=0.1),
        same_on_batch=False
    )
    
    # Training of the model
    for epoch in tqdm(range(epochs), total=epochs):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(trainloader, 0):
            # Limiting training data for faster epochs.
            if batch_idx * batch_size >= n_train_examples:
                break
            
            inputs = augmentations(inputs).to(device)
            labels = labels.to(device)  # Flatten to 1D if needed

            output_vals = model.forward(inputs)
            
            loss = loss_criterion(output_vals, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation of the model
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(validloader):
                # Limiting validation data.
                if batch_idx * batch_size >= n_valid_examples:
                    break
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                output_vals = model(inputs)
                # Get the index of the max log-probability.
                pred = output_vals.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()

        accuracy = correct / min(len(validloader.dataset), n_valid_examples)

        # report back to Optuna how far it is (epoch-wise) into the trial and how well it is doing (accuracy)
        trial.report(accuracy, epoch)
        
        # Step the scheduler
        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(accuracy)  # Use validation accuracy for the plateau scheduler
        else:
            scheduler.step() 

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


def save_best_params(best_params, model_name, loss_value, base_dir='checkpoints/optuna_params'):
    """Save the best parameters to a structured directory."""
    # Create the base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Create a model-specific directory
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Create the filename
    filename = f"{model_name}_{loss_value:.4f}.json"
    file_path = os.path.join(model_dir, filename)

    # Save the parameters to a JSON file
    with open(file_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print(f"Best parameters saved to {file_path}")

def optuna_param_search(model_name, loss_criterion, num_epochs_for_experiments=12, device='cpu', transfer_learning=False):
    
    print(f"Optuna is done on device: {device}")
    
    objective_with_args = partial(objective, model_name=model_name, epochs=num_epochs_for_experiments, device=device, loss_criterion=loss_criterion)
    
    # make the study
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name="mri-alzhimer-classification", direction="maximize", sampler=sampler)
    study.optimize(objective_with_args, n_trials=40)

    # get the purned and completed trials
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # print statistics
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    save_best_params(trial.params, model_name, trial.value)
        
    return trial.params