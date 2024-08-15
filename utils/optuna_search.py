import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial
import optuna
from torch import optim

from models import def_models
  
def define_model(trial, model_name):
    
    models = ['NaiveModelAug' , 'NaiveModel']
    
    # Check if the model name is one of the models in the list
    if model_name not in models:
        raise ValueError(f"Model name {model_name} is not in the list of available models: {models}")
    
    input_channels = 3
    output_channels = 4
    
    # Define the model based on the name
    model_class = getattr(def_models, model_name)
    model = model_class(input_channels, output_channels)
    
    return model
    

    

def objective(trial,model_name, epochs, device , loss_criterion):

    # Generate the model
    model = define_model(trial, model_name).to(device)
    
    
    # Hyperparameters to experiment : learning rate, optimizer, batch size
    # learning rate
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)  # log=True, will use log scale to interplolate between lr
    # optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    # batch size
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    

    # Get MRI dataset - load the data and shuffle it
    train_set = torch.load('dataset/dataset_variables/train_set.pt')
    validation_set = torch.load('dataset/dataset_variables/validation_set.pt')
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    
    # limit train examples
    n_train_examples = 50 * batch_size
    n_valid_examples = 15 * batch_size
    
    
    # Training of the model
    for epoch in tqdm(range(epochs), total=epochs):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            
            # Limiting training data for faster epochs.
            if batch_idx * batch_size >= n_train_examples:
                break

            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            loss = loss_criterion(output, labels)
            
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
                
                output = model(inputs)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()

        accuracy = correct / min(len(validloader.dataset), n_valid_examples)

        # report back to Optuna how far it is (epoch-wise) into the trial and how well it is doing (accuracy)
        trial.report(accuracy, epoch)  

        # then, Optuna can decide if the trial should be pruned
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


def optuna_param_search(model_name, loss_criterion, num_epochs_for_experiments=10, device='cpu'):
    
    objective_with_args = partial(objective, 
                                  model_name=model_name, 
                                  epochs=num_epochs_for_experiments, 
                                  device=device, 
                                  loss_criterion=loss_criterion)
    
    # make the study
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name="mri-alzhimer-classification", direction="maximize", sampler=sampler)
    study.optimize(objective_with_args, n_trials=3, timeout=2000)

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
        
    return trial.params
    




