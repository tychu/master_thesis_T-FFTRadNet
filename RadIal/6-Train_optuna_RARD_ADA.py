import os
import json
import argparse
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from model.FFTRadNet_ViT_RARD_ADA import FFTRadNet_ViT
from model.FFTRadNet_ViT_RARD_ADA import FFTRadNet_ViT_ADC
from dataset.dataset import RADIal
from dataset.matlab_dataset_RARD import MATLAB
from dataset.encoder_RARD import ra_encoder
from dataset.dataloader_RARD import CreateDataLoaders
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from loss.loss_RARD import pixor_loss, MVloss
from utils.evaluation_RARD import run_evaluation
import torch.nn as nn
import matplotlib.pyplot as plt

import optuna
from optuna.trial import TrialState
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback



# Training function
def train(config, net, train_loader, optimizer, scheduler, history, kbar):
    """
    Perform one epoch of training for the model.

    Args:
        config (dict): Configuration dictionary.
        net (torch.nn.Module): Neural network to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for the model.
        scheduler (Scheduler): Learning rate scheduler.
        history (dict): Dictionary to store training history.
        kbar (Kbar): Progress bar for the training process.

    Returns:
        Tuple: Losses and model outputs for the epoch.
    """
    net.train()  # Set the model to training mode

    # Initialize running losses
    running_loss = 0.0
    running_cls_loss, running_reg_loss, running_mv_loss = 0.0, 0.0, 0.0
    running_cls_loss_ra, running_cls_loss_rd = 0.0, 0.0

    # Loop through the training data
    for i, data in enumerate(train_loader):
        # Prepare inputs and labels based on data mode
        if config['data_mode'] == 'ADC':
            inputs = data[0].to('cuda').type(torch.complex64)
        else:
            inputs = data[0].to('cuda').float()

        label_ra_map = data[1].to('cuda').float()
        label_rd_map = data[2].to('cuda').float()

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        with torch.set_grad_enabled(True):
            outputs = net(inputs)

        # Calculate losses for RA and RD maps
        classif_loss_ra, reg_loss_ra = pixor_loss(outputs['RA'], label_ra_map, config['losses'])
        classif_loss_rd, reg_loss_rd = pixor_loss(outputs['RD'], label_rd_map, config['losses'])

        # Combine losses
        classif_loss = classif_loss_ra + classif_loss_rd
        reg_loss = reg_loss_ra + reg_loss_rd
        mvloss = MVloss(outputs)

        # Update running losses
        running_mv_loss += mvloss.item() * inputs.size(0)
        running_cls_loss_ra += classif_loss_ra.item() * inputs.size(0)
        running_cls_loss_rd += classif_loss_rd.item() * inputs.size(0)
        running_cls_loss = running_cls_loss_ra + running_cls_loss_rd
        running_reg_loss += reg_loss.item() * inputs.size(0)

        # Apply weights to losses
        classif_loss *= config['losses']['weight'][0]
        reg_loss *= config['losses']['weight'][1]
        mvloss *= config['losses']['weight'][2]

        # Calculate total loss and perform backpropagation
        loss = classif_loss + reg_loss + mvloss
        loss.backward()
        optimizer.step()

        # Update running total loss
        running_loss += loss.item() * inputs.size(0)

    # Update the learning rate scheduler
    scheduler.step()

    # Record training history
    history['train_loss'].append(running_loss / len(train_loader.dataset))
    history['lr'].append(scheduler.get_last_lr()[0])

    return (
        running_cls_loss_ra / len(train_loader.dataset),
        running_cls_loss_rd / len(train_loader.dataset),
        running_loss / len(train_loader.dataset),
        outputs,
        running_cls_loss / len(train_loader.dataset),
        running_reg_loss / len(train_loader.dataset),
        mvloss / len(train_loader.dataset)
    )
    

# Optuna objective function
def objective(trial, config, resume):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (Trial): Optuna trial object.
        config (dict): Configuration dictionary.
        resume (str): Path to checkpoint file for resuming training.

    Returns:
        float: Final training loss or other evaluation metric.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # Generate a unique experiment name
    curr_date = datetime.now()
    exp_name = f"{config['name']}___{curr_date.strftime('%b-%d-%Y___%H:%M:%S')}"

    # Create output directories
    output_folder = Path(config['output']['dir'])
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / exp_name).mkdir(parents=True, exist_ok=True)

    # Save the configuration file for reference
    with open(output_folder / exp_name / 'config.json', 'w') as outfile:
        json.dump(config, outfile)

    # Set the device for training
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    # Hyperparameter suggestions from Optuna
    optuna_para_config = {
        "optimizer": {
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "step_size": trial.suggest_int('step_size', 5, 15, step=5)
        },
        "batch_size": trial.suggest_categorical('batch_size', [4, 8])
    }

    # Initialize the encoder
    enc = ra_encoder(
        geometry=config['dataset']['geometry'],
        statistics=config['dataset']['statistics'],
        regression_layer=2
    )

    # Initialize the model based on data mode
    if config['data_mode'] != 'ADC':
        net = FFTRadNet_ViT(
            patch_size=config['model']['patch_size'],
            channels=config['model']['channels'],
            in_chans=config['model']['in_chans'],
            embed_dim=config['model']['embed_dim'],
            depths=config['model']['depths'],
            num_heads=config['model']['num_heads'],
            drop_rates=config['model']['drop_rates'],
            regression_layer=2,
            detection_head=config['model']['DetectionHead']
        )
        dataset = MATLAB(
            root_dir=config['dataset']['root_dir'],
            folder_dir=config['dataset']['data_folder'],
            statistics=config['dataset']['statistics'],
            encoder=enc.encode,
            perform_FFT=config['data_mode']
        )
    else:
        net = FFTRadNet_ViT_ADC(
            patch_size=config['model']['patch_size'],
            channels=config['model']['channels'],
            in_chans=config['model']['in_chans'],
            embed_dim=config['model']['embed_dim'],
            depths=config['model']['depths'],
            num_heads=config['model']['num_heads'],
            drop_rates=config['model']['drop_rates'],
            regression_layer=2,
            detection_head=config['model']['DetectionHead'],
            segmentation_head=config['model']['SegmentationHead']
        )
        dataset = MATLAB(
            root_dir=config['dataset']['root_dir'],
            folder_dir=config['dataset']['data_folder'],
            statistics=config['dataset']['statistics'],
            encoder_ra=enc.ra_encode,
            encoder_rd=enc.rd_encode,
            perform_FFT='ADC'
        )

    # Count total trainable parameters
    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ", t_params)
    net.to(device)

    # Set optimizer and scheduler
    lr = optuna_para_config['optimizer']['lr']
    step_size = optuna_para_config['optimizer']['step_size']
    gamma = float(config['lr_scheduler']['gamma'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Determine batch size and number of epochs
    batch_size = optuna_para_config['batch_size']
    num_epochs = 100 if batch_size in [4, 8] else 200

    # Initialize training history
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'mAP': [], 'mAR': [], 'val_f1': [], 'train_f1': []}

    # Create DataLoaders
    train_loader, val_loader, _ = CreateDataLoaders(dataset, batch_size, config['dataloader'], config['seed'])

    # Initialize Weights and Biases for tracking
    wandb.init(
        project=config['optuna_project'],
        entity="chu06-imec",
        config=dict(trial.params),
        group='TFFTRadNet_optimization',
        reinit=True
    )

    # Resume training if required
    if resume and trial.number == 0:
        print('===========  Resume training ==================')
        cp_dict = torch.load(resume)
        net.load_state_dict(cp_dict['net_state_dict'])
        optimizer.load_state_dict(cp_dict['optimizer'])
        scheduler.load_state_dict(cp_dict['scheduler'])

    # Training loop
    for epoch in range(num_epochs):
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs)

        # Train for one epoch
        cls_loss_ra, cls_loss_rd, loss, predictions, cls_loss, reg_loss, mv_loss = train(
            config, net, train_loader, optimizer, scheduler, history, kbar
        )

        # Validate the model
        eval = run_evaluation(
            net, val_loader, enc, check_perf=(epoch >= 1),
            detection_loss=pixor_loss, losses_params=config['losses'], config=config
        )

        # Update training history
        history['val_loss'].append(eval['loss'] / len(val_loader.dataset))
        history['mAP'].append(eval['mAP'])
        history['mAR'].append(eval['mAR'])

        # Calculate F1 score
        F1_score = (eval['mAP'] * eval['mAR']) / ((eval['mAP'] + eval['mAR']) / 2) if eval['mAP'] + eval['mAR'] > 0 else 0
        history['val_f1'].append(F1_score)

        # Log metrics to Weights and Biases
        wandb.log({
            "validation F1 score": F1_score,
            "validation precision": eval['mAP'],
            "validation recall": eval['mAR'],
            "Validation loss": eval['loss'] / len(val_loader.dataset),
            "Training loss": loss,
        }, step=epoch)

        # Check for trial pruning
        trial.report(F1_score, epoch)
        if trial.should_prune():
            wandb.run.summary["state"] = "pruned"
            wandb.finish(quiet=True)
            raise optuna.exceptions.TrialPruned()

        # Save model checkpoint
        if epoch >= 20:
            checkpoint = {
                'net_state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'batch_size': batch_size,
                'lr': lr,
                'step_size': step_size,
                'history': history,
                'detectionhead_output': predictions
            }
            torch.save(checkpoint, output_folder / exp_name / f"epoch_{epoch}.pth")

    # Finalize Weights and Biases
    wandb.run.summary["state"] = "completed"
    wandb.finish(quiet=True)

    return loss

# Main execution
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='FFTRadNet Training with Optuna')
    parser.add_argument('-c', '--config', default='config.json', type=str, help='Path to the config file')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('-r', '--resume', default=None, type=str, help='Path to checkpoint to resume training')
    args = parser.parse_args()

    # Load configuration
    config = json.load(open(args.config))
    os.environ["WANDB_START_METHOD"] = "thread"

    # Define fixed parameters for the baseline trial
    fixed_params = {"lr": 1e-4, "step_size": 10, "batch_size": 8}
    fixed_trial = optuna.trial.FixedTrial(fixed_params)

    # Run baseline trial
    baseline_score = objective(fixed_trial, config, args.resume)

    # Create Optuna study
    study = optuna.create_study(direction='maximize', study_name='FFTRadNet_optimization', load_if_exists=True)
    study.add_trial(optuna.create_trial(
        state=TrialState.COMPLETE, value=baseline_score, params=fixed_params,
        distributions={
            "lr": optuna.distributions.FloatDistribution(1e-4, 5e-3, log=True),
            "step_size": optuna.distributions.IntDistribution(5, 20, step=5),
            "batch_size": optuna.distributions.IntDistribution(4, 8, step=4)
        }
    ))

    # Optimize the objective function
    study.optimize(lambda trial: objective(trial, config, args.resume), n_trials=args.trials)

    # Save study results
    with open('optuna_results.txt', 'w') as f:
        f.write(f"Best trial: {study.best_trial}\n")
    study.trials_dataframe().to_csv('optuna_study.csv')
