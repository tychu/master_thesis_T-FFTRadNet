import os
import json
import argparse
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
#from torch.utils.tensorboard import SummaryWriter

from model.FFTRadNet_ViT_RARD import FFTRadNet_ViT
from model.FFTRadNet_ViT_RARD import FFTRadNet_ViT_ADC

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



def train(config, net, train_loader, optimizer, scheduler, history, kbar):
    net.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_reg_loss = 0.0
    running_mv_loss = 0.0
    running_cls_loss_ra = 0.0
    running_cls_loss_rd = 0.0

    for i, data in enumerate(train_loader):
        if config['data_mode'] == 'ADC':
                inputs = data[0].to('cuda').type(torch.complex64)
        else:
            inputs = data[0].to('cuda').float()
        #inputs = data[0].to('cuda').float()

        label_ra_map = data[1].to('cuda').float() # matlab_dataset.py: out_label_ra
        label_rd_map = data[2].to('cuda').float() # matlab_dataset.py: out_label_rd

        # reset the gradient
        optimizer.zero_grad()

        # forward pass, enable to track our gradient
        with torch.set_grad_enabled(True):
            outputs = net(inputs)

        classif_loss_ra, reg_loss_ra = pixor_loss(outputs['RA'], label_ra_map, config['losses'])
        #print("RD loss")
        classif_loss_rd, reg_loss_rd = pixor_loss(outputs['RD'], label_rd_map, config['losses'])
        classif_loss = classif_loss_ra + classif_loss_rd
        reg_loss = reg_loss_ra + reg_loss_rd*3

        mvloss = MVloss(outputs)
        running_mv_loss += mvloss.item() * inputs.size(0)
        #print("classif_loss: ", classif_loss, "reg_loss: ", reg_loss, "mvloss: ", mvloss)
        #print("inputs.size(0): ", inputs.size(0))
        #running_cls_loss += classif_loss.item() * inputs.size(0)
        running_cls_loss_ra += classif_loss_ra.item() * inputs.size(0)
        running_cls_loss_rd += classif_loss_rd.item() * inputs.size(0)
        running_reg_loss += reg_loss.item() * inputs.size(0)
        #print("running_cls_loss: ", running_cls_loss, "running_reg_loss: ", running_reg_loss)
        #print("classif_loss: ", classif_loss, "reg_loss: ", reg_loss)

        classif_loss *= config['losses']['weight'][0]
        reg_loss *= config['losses']['weight'][1]
        mvloss *= config['losses']['weight'][2]
        #print("mv loss weight: ", config['losses']['weight'][2])
        #print("after weight classif_loss: ", classif_loss, "reg_loss: ", reg_loss)
        loss = classif_loss + reg_loss + mvloss
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
        #print("running_loss: ", running_loss, "loss: ", loss)

        # if i >= 2:
        #     raise Exception("finishing checking loss")

        

        #kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), ("reg", reg_loss.item()) ] )

    scheduler.step()
    history['train_loss'].append(running_loss / len(train_loader.dataset))
    history['lr'].append(scheduler.get_last_lr()[0])
    #print("len(train_loader.dataset): ", len(train_loader.dataset))

    return running_cls_loss_ra/ len(train_loader.dataset), running_cls_loss_rd/ len(train_loader.dataset), running_loss / len(train_loader.dataset), outputs, running_cls_loss/ len(train_loader.dataset), running_reg_loss/ len(train_loader.dataset), mvloss/ len(train_loader.dataset)

    

def objective(trial, config, resume):
    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # create experience name
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    
    # Create directory structure
    output_folder = Path(config['output']['dir'])
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / exp_name).mkdir(parents=True, exist_ok=True)

    with open(output_folder / exp_name / 'config.json', 'w') as outfile:
        json.dump(config, outfile)

    # set device
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    # Set up the config as per trial parameters
    optuna_para_config = {
        "optimizer": {
            "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True), # original: 1-e4
            "step_size": trial.suggest_int('step_size', 5, 15, step=5) # original: 10
        },
        #  "model": {
        #       "embed_dim": trial.suggest_categorical('embed_dim', [24, 48]), 
        # #     "mimo_layer": trial.suggest_int('mimo_layer', 64, 192, step=64) # original: 192
        #  },
        #"batch_size": trial.suggest_categorical('batch_size', [4, 8]), # original: 4
        #"threshold":  trial.suggest_float("FFT_confidence_threshold", 0.1, 0.2, step=0.05) # original: 0.2
    }
    #print("config['dataset']['geometry']: ", config['dataset']['geometry'])
    enc = ra_encoder(geometry=config['dataset']['geometry'],
                     statistics=config['dataset']['statistics'],
                     regression_layer=2)

    # Create the model
    # net = FFTRadNet(
    #     blocks=config['model']['backbone_block'],
    #     mimo_layer= optuna_para_config['model']['mimo_layer'],
    #     Ntx = config['model']['NbTxAntenna'],
    #     Nrx = config['model']['NbRxAntenna'],
    #     channels=config['model']['channels'],
    #     regression_layer=2,
    #     detection_head=config['model']['DetectionHead']
    # )


    # Create the model
    if config['data_mode'] != 'ADC':
        net = FFTRadNet_ViT(patch_size = config['model']['patch_size'],
                        channels = config['model']['channels'],
                        in_chans = config['model']['in_chans'], # Number of input image channels, paper set: 32 (rx*2)
                        #embed_dim = optuna_para_config['model']['embed_dim'], 
                        embed_dim = config['model']['embed_dim'], # linear embedding paper set: 48
                        depths = config['model']['depths'],
                        num_heads = config['model']['num_heads'],
                        drop_rates = config['model']['drop_rates'],
                        regression_layer = 2,
                        detection_head = config['model']['DetectionHead'])

        # dataset = RADIal(root_dir = config['dataset']['root_dir'],
        #                     statistics= config['dataset']['statistics'],
        #                     encoder=enc.encode,
        #                     difficult=True,perform_FFT=config['data_mode'])

        dataset = MATLAB(root_dir = config['dataset']['root_dir'],
                         folder_dir = config['dataset']['data_folder'],
                            statistics= config['dataset']['statistics'],
                            encoder=enc.encode,perform_FFT=config['data_mode'])

    else:
        net = FFTRadNet_ViT_ADC(patch_size = config['model']['patch_size'],
                        channels = config['model']['channels'],
                        in_chans = config['model']['in_chans'], # Number of input image channels, paper set: 32 (rx*2)
                        embed_dim = config['model']['embed_dim'],
                        depths = config['model']['depths'],
                        num_heads = config['model']['num_heads'],
                        drop_rates = config['model']['drop_rates'],
                        regression_layer = 2, # predict range, angle
                        detection_head = config['model']['DetectionHead'],
                        segmentation_head = config['model']['SegmentationHead'])

        # dataset = RADIal(root_dir = config['dataset']['root_dir'],
        #                     statistics= config['dataset']['statistics'],
        #                     encoder=enc.encode,
        #                     difficult=True,perform_FFT='ADC')
        dataset = MATLAB(root_dir = config['dataset']['root_dir'],
                         folder_dir = config['dataset']['data_folder'],
                            statistics= config['dataset']['statistics'],
                            encoder_ra=enc.ra_encode, encoder_rd=enc.rd_encode,
                            perform_FFT='ADC')
        
    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    net.to('cuda')
    #print(net)

    # Define hyperparameters to be tuned
    lr = optuna_para_config['optimizer']['lr']
    step_size = optuna_para_config['optimizer']['step_size']
    gamma =  float(config['lr_scheduler']['gamma']) #trial.suggest_uniform('gamma', 0.1, 0.9)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # set num_epochs
    # num_epochs = int(config['num_epochs'])
    #batch_size = config['dataloader']['train']['batch_size'] # 
    batch_size = 4 #optuna_para_config['batch_size']
    if batch_size == 4 or batch_size == 8:
        num_epochs = 100# 150
    elif batch_size == 16 or batch_size == 32:
        num_epochs = 200
    
    
    threshold = 0.2 #trial.suggest_categorical('threshold', [0.2]) # paper set 0.2
    
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'mAP': [], 'mAR': [], 'val_f1': [], 'train_f1': []}

    train_loader, val_loader, _ = CreateDataLoaders(dataset, batch_size, config['dataloader'], config['seed'])
    
    # init tracking experiment.
    # hyper-parameters, trial id are stored.
    config_optuna = dict(trial.params)
    config_optuna["trial.number"] = trial.number
    wandb.init(
        project=config['optuna_project'],
        entity="chu06-imec",  # NOTE: this entity depends on your wandb account.
        config=config_optuna,
        group='TFFTRadNet_optimization',
        reinit=True,
    )

    if resume:
        print('===========  Resume training  ==================:')
        cp_dict = torch.load(resume)
        net.load_state_dict(cp_dict['net_state_dict'])
        optimizer.load_state_dict(cp_dict['optimizer'])
        scheduler.load_state_dict(cp_dict['scheduler'])
        #startEpoch = cp_dict['epoch']+1
        #history = cp_dict['history']
        #global_step = cp_dict['global_step']


    for epoch in range(num_epochs): # num_epochs
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

        cls_loss_ra, cls_loss_rd, loss,predictions, cls_loss, reg_loss, mv_loss = train(config, net, train_loader, optimizer, scheduler, history, kbar)
        #raise Exception("finishing checking one epoch")
        
        # tra = run_evaluation(trial, net, train_loader, enc, threshold, check_perf=(epoch >= 1),
        #                       detection_loss=pixor_loss, segmentation_loss=None,
        #                       losses_params=config['losses'])
        
        eval = run_evaluation(net, val_loader, enc, check_perf= (epoch >= 1),
                               detection_loss=pixor_loss, 
                               losses_params=config['losses'], config=config)
        history['val_loss'].append(eval['loss']/ len(val_loader.dataset))
        history['mAP'].append(eval['mAP'])
        history['mAR'].append(eval['mAR'])

        if eval['mAP'] + eval['mAR'] == 0:
            F1_score = 0
        else:
            F1_score = (eval['mAP']*eval['mAR'])/((eval['mAP'] + eval['mAR'])/2)
        

        
        history['val_f1'].append(F1_score)

        #Pruning
        trial.report(F1_score, epoch)
        # report F1_score to wandb
        wandb.log(data={"validation F1 score": F1_score, 
                        "validation precision":eval['mAP'], 
                        "validation recall":eval['mAR'], 
                        "Validation loss":eval['loss']/ len(val_loader.dataset),
                        "Validation classification loss": eval['running_cls_loss']/ len(val_loader.dataset), 
                        "Validation regression loss": eval['running_reg_loss']/ len(val_loader.dataset), 
                        "Validation MV loss": eval['running_mv_loss']/ len(val_loader.dataset),
                        "Validation IoU RD":eval['miou_RD'],

                        "Training loss":loss, 
                        "Training classification loss": cls_loss,
                        "Training RA classification loss": cls_loss_ra,
                        "Training RD classification loss": cls_loss_rd, 
                        "Training regression loss": reg_loss, 
                        "Training MV loss": mv_loss,
                        "Training MV weight loss":mv_loss*config['losses']['weight'][2]
                                                  }, 
                        step=epoch)

        if trial.should_prune():
            wandb.run.summary["state"] = "pruned"
            wandb.finish(quiet=True)
            raise optuna.exceptions.TrialPruned()

        name_output_file = config['name']+'_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}_trialnumber_{:02d}_batch{:02d}.pth'.format(epoch, loss, eval['mAP'], eval['mAR'], trial.number, batch_size)
        filename = output_folder / exp_name / name_output_file

        checkpoint={}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['batch_size'] = batch_size
        checkpoint['lr'] = lr
        checkpoint['step_size'] = step_size
        checkpoint['history'] = history
        checkpoint['detectionhead_output'] = predictions

        if epoch >= 60:
            torch.save(checkpoint,filename)
            print('')
    
    # report the final validation accuracy to wandb
    #wandb.run.summary["final accuracy"] = F1_score
    wandb.run.summary["state"] = "completed"
    wandb.finish(quiet=True)



    return loss #F1_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FFTRadNet Training with Optuna')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    args = parser.parse_args()
    config = json.load(open(args.config))

    # wandb might cause an error without this.
    os.environ["WANDB_START_METHOD"] = "thread"

    # Specific parameter combination for the first trial
    fixed_params = {
        "lr": 1e-3,
        "step_size": 10,
        #"batch_size": 4,
        #"embed_dim": 48
    }

    # Create a FixedTrial object
    fixed_trial = optuna.trial.FixedTrial(fixed_params)

    # Evaluate the objective function with the fixed parameters
    baseline_score = objective(fixed_trial, config, args.resume)

    # Create a study
    # optuna without parallel
    study = optuna.create_study(direction='maximize', study_name='FFTRadNet_optimization', 
                                #storage = "postgresql://chu06:hello_psql@localhost:35931/mydb",
                                load_if_exists = True,
                                pruner=optuna.pruners.PercentilePruner(50.0, n_startup_trials=3,
                                           n_warmup_steps=30, interval_steps=10))
    
    study = optuna.create_study(direction='maximize')

    # Add the baseline trial to the study
    study.add_trial(optuna.create_trial(
        state=optuna.trial.TrialState.COMPLETE,
        value=baseline_score,
        params=fixed_params,
        distributions={
            "lr": optuna.distributions.FloatDistribution(1e-4, 1e-3, log=True),
            "step_size": optuna.distributions.IntDistribution(5, 15, step=5),
            #"embed_dim": optuna.distributions.IntDistribution(24, 48, step=24), 
            #"batch_size": optuna.distributions.IntDistribution(4, 8, step=4),
        }
    ))
 
    study.optimize(lambda trial: objective(trial, config, args.resume), n_trials=args.trials)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    trial = study.best_trial
    output_file = 'optuna_paramter_tuning'
    # Open the file in append mode and write the required information
    with open(output_file, 'a') as f:
        f.write("Study statistics:\n")
        f.write(f"  Number of finished trials: {len(study.trials)}\n")
        f.write(f"  Number of pruned trials: {len(pruned_trials)}\n")
        f.write(f"  Number of complete trials: {len(complete_trials)}\n")
        f.write("Best trial:\n")
        f.write(f"  Value: {trial.value}\n")
        f.write("  Params: \n")
        for key, value in trial.params.items():
            f.write(f"    {key}: {value}\n")

    # Optionally save the study
    study.trials_dataframe().to_csv('optuna_study.csv')
