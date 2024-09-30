import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial
import optuna
from torch import optim
import torchvision.transforms as transforms
import json

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
    
    models = ['NaiveModelAug', 'NaiveModel', 'DINO_v2_FT']
    
    if model_name not in models:
        raise ValueError(f"Model name {model_name} is not in the list of available models: {models}")
    
    output_channels = 4  

    model_class = getattr(def_models, model_name)
    
    if model_name == 'DINO_v2_FT':
        # Load the DINO v2 backbone
        dino_v2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # Freeze the DINO v2 layers if using transfer learning
        if transfer_learning:
            for param in dino_v2_model.parameters():
                param.requires_grad = False
        
        # Create the DINO_v2_FT model, passing the backbone
        model = model_class(dino_backbone=dino_v2_model, output_channels=output_channels)
    else:
        # Handle other model types (NaiveModel, NaiveModelAug) as needed
        model = model_class(input_channels=3, output_channels=output_channels)

    return model
    

    

def objective(trial,model_name, epochs, device , loss_criterion, transfer_learning=True):

    # Generate the model
    model = define_model(trial, model_name, transfer_learning).to(device)
    
    # Hyperparameters to experiment : learning rate, optimizer, batch size
    # learning rate
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)  # log=True, will use log scale to interplolate between lr
    # optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    # batch size
    batch_size = trial.suggest_categorical('batch_size', [32,64,128,256])
    

    # Get MRI dataset - load the data and shuffle it
    train_set = torch.load('dataset/dataset_variables/train_set.pt')
    validation_set = torch.load('dataset/dataset_variables/validation_set.pt')
    
    
    if (type(model).__name__ == 'DINO_v2_FT'):
        # Resize images from 128x128 to 224x224
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor()
        ])

        # Apply transformations to the datasets
        train_set = utils_funcs.apply_transformations(train_set, preprocess)
        validation_set = utils_funcs.apply_transformations(validation_set, preprocess)
    
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3)
    validloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=3)
    
    # limit train examples
    n_train_examples = 50 * batch_size
    n_valid_examples = 15 * batch_size
    
    # Training of the model
    for epoch in tqdm(range(epochs), total=epochs):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(trainloader, 0):
            
            # Limiting training data for faster epochs.
            if batch_idx * batch_size >= n_train_examples:
                break
            
            inputs = inputs.to(device)
            labels = labels.to(device)

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

        # then, Optuna can decide if the trial should be pruned
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

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

def optuna_param_search(model_name, loss_criterion, num_epochs_for_experiments=10, device='cpu', transfer_learning=False):
    
    print(f"Optuna is done on device: {device}")
    
    objective_with_args = partial(objective, 
                                  model_name=model_name, 
                                  epochs=num_epochs_for_experiments, 
                                  device=device, 
                                  loss_criterion=loss_criterion)
    
    # make the study
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name="mri-alzhimer-classification", direction="maximize", sampler=sampler)
    study.optimize(objective_with_args, n_trials=30)

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
    




