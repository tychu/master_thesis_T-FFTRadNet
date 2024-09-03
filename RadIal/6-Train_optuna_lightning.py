import os
import json
import argparse
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime

from model.FFTRadNet_ViT_ddp import FFTRadNet_ViT
from model.FFTRadNet_ViT_ddp import FFTRadNet_ViT_ADC

from dataset.dataset import RADIal
from dataset.matlab_dataset_ddp import MATLAB
#from dataset.encoder_gpu import ra_encoder
from dataset.encoder_modi import ra_encoder # cpu
from dataset.dataloader_gpu import CreateDataLoaders
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from loss import pixor_loss
from utils.evaluation_ddp import run_evaluation_lightning
import torch.nn as nn
import matplotlib.pyplot as plt

import optuna
from optuna.trial import TrialState
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback

from optuna.integration  import PyTorchLightningPruningCallback
import lightning.pytorch  as pl
from lightning.pytorch import Trainer

from lightning.pytorch.loggers import WandbLogger

#from utils.metrics_gpu import GetFullMetrics, Metrics
from utils.metrics_ddp import GetFullMetrics, Metrics # cpu

from lightning.pytorch.strategies import DDPStrategy

import time

from lightning.pytorch.callbacks import ModelCheckpoint
import socket
from lightning.fabric import Fabric

class LightningNet(pl.LightningModule):
    def __init__(self, config, step_size, lr, history, encoder, data_module, config_optuna):
        super(LightningNet, self).__init__()
        #self.model = Net(dropout, output_dims)
        self.config = config
        self.lr = lr
        self.step_size = step_size
        self.history = history
        self.encoder = encoder
        self.metrics = Metrics()
        self.val_dataloader = data_module.val_dataloader()
        self.val_step_outputs = []
        self.config_optuna = config_optuna

        self.save_hyperparameters()  # This saves hyperparameters to the checkpoint

        if config['data_mode'] != 'ADC':
            self.net = FFTRadNet_ViT(patch_size = config['model']['patch_size'],
                            channels = config['model']['channels'],
                            in_chans = config['model']['in_chans'], # Number of input image channels, paper set: 32 (rx*2)
                            #embed_dim = optuna_para_config['model']['embed_dim'], 
                            embed_dim = config['model']['embed_dim'], # linear embedding paper set: 48
                            depths = config['model']['depths'],
                            num_heads = config['model']['num_heads'],
                            drop_rates = config['model']['drop_rates'],
                            regression_layer = 2,
                            detection_head = config['model']['DetectionHead'])

        else:
            self.net = FFTRadNet_ViT_ADC(patch_size = config['model']['patch_size'],
                            channels = config['model']['channels'],
                            in_chans = config['model']['in_chans'], # Number of input image channels, paper set: 32 (rx*2)
                            embed_dim = config['model']['embed_dim'],
                            depths = config['model']['depths'],
                            num_heads = config['model']['num_heads'],
                            drop_rates = config['model']['drop_rates'],
                            regression_layer = 2,
                            detection_head = config['model']['DetectionHead'],
                            segmentation_head = config['model']['SegmentationHead'])
        t_params = sum(p.numel() for p in self.net.parameters())
        print("Network Parameters: ",t_params)
        #self.net.to('cuda')
        #print(net)
        
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        print("training_step .....")
        inputs, label_map, _ = batch
        if self.config['data_mode'] == 'ADC':
            #inputs = inputs.to(self.device).type(torch.complex64)
            inputs = inputs.type(torch.complex64)
        else:
            #inputs = inputs.to(self.device).float()
            inputs = inputs.float()
        
        #label_map = label_map.to(self.device).float()
        label_map = label_map.float()

        outputs = self(inputs)
        classif_loss, reg_loss = pixor_loss(outputs, label_map, self.config['losses'])
        classif_loss *= self.config['losses']['weight'][0]
        reg_loss *= self.config['losses']['weight'][1]
        loss = classif_loss + reg_loss

        self.log("train_loss", loss, sync_dist=True)
        #print("wandb log in train: ", loss)
        #wandb.log(data={"train_loss": loss})
        

        return loss
  
    # def on_train_epoch_start(self):
    #     """Called when the train epoch begins."""
    #     self.start_time = time.time()  # Record the start time of the epoch

    def on_train_epoch_end(self):
        mAP,mAR, mIoU, TP, FP, FN = self.metrics.GetMetrics()
        if mAP + mAR == 0:
            F1_score = 0
        else:
            F1_score = (mAP*mAR)/((mAP + mAR)/2)

        self.log("val_precision", mAP, sync_dist=True)
        self.log("val_recall", mAR, sync_dist=True)
        self.log("f1_score", F1_score, sync_dist=True)
        #wandb.log(data={"val_precision": mAP})
        #wandb.log(data={"val_recall": mAR})
        #wandb.log(data={"f1_score": F1_score})

        

    def validation_step(self, batch, batch_idx):
        
        print("validation_step ...")
        inputs, label_map, labels = batch

        if self.config['data_mode'] == 'ADC':
            #inputs = inputs.to(self.device).type(torch.complex64)
            inputs = inputs.type(torch.complex64)
        else:
            #inputs = inputs.to(self.device).float()
            inputs = inputs.float()
        
        #label_map = label_map.to(self.device).float()
        label_map = label_map.float()

        outputs = self(inputs)
        classif_loss, reg_loss = pixor_loss(outputs, label_map, self.config['losses'])
        classif_loss *= self.config['losses']['weight'][0]
        reg_loss *= self.config['losses']['weight'][1]
        loss = classif_loss + reg_loss

        self.log("val_loss", loss, sync_dist=True)
        #wandb.log(data={"val_loss": loss})

        
        

        if self.current_epoch >= 1:
            out_obj = outputs.detach().cpu().numpy().copy()
            
            
            #print("save validation loss: ", loss)
            for pred_obj,true_obj in zip(out_obj, labels):
                pred_map = 0 # not used in update() for computing pred_map=PredMap
                true_map = 0 # not used in update() for computing ture_map=lable_map
                true_obj = true_obj.detach().cpu().numpy().copy()

                #metrics.update(pred_map[0],true_map,np.asarray(encoder.decode(pred_obj,0.05)),true_obj,
                #            threshold=0.2,range_min=5,range_max=100) 
                #start_time = time.time()
                #print("cpu: ", start_time)
                self.metrics.update(pred_map,true_map,np.asarray(self.encoder.decode(pred_obj,0.05)),true_obj,
                            threshold=0.2,range_min=0,range_max=345) 
                #end_time = time.time()
                #computation_time = end_time - start_time
                #print(f"Computation time = {computation_time:.4f} seconds")

        



    def configure_optimizers(self):
        gamma = float(self.config['lr_scheduler']['gamma'])

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=gamma)
        return [optimizer], [scheduler]
    
    def prepare_data(self):
        #  ONLY WHEN LOCAL_RANK = 0
        print(".......os.environ.get('LOCAL_RANK') == '0' .....")
        # Initialize W&B
        wandb.init(
        project=config['optuna_project'],
        entity="chu06-imec",  # NOTE: this entity depends on your wandb account.
        config=self.config_optuna,
        group='TFFTRadNet_optimization',
        reinit=True,
        )

    

    
class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super(DataModule, self).__init__()
        self.config = config
        self.train_loader = None
        self.val_loader = None

    def setup(self, stage=None):
        # Setup data transformation or preprocessing here if necessary

        # Initialize your datasets here
        enc = ra_encoder(geometry=self.config['dataset']['geometry'],
                         statistics=self.config['dataset']['statistics'],
                         regression_layer=2)
        
        if self.config['data_mode'] != 'ADC':
            dataset = MATLAB(
                root_dir=self.config['dataset']['root_dir'],
                folder_dir=self.config['dataset']['data_folder'],
                statistics=self.config['dataset']['statistics'],
                encoder=enc.encode,
                perform_FFT=self.config['data_mode']
            )
        else:
            dataset = MATLAB(
                root_dir=self.config['dataset']['root_dir'],
                folder_dir=self.config['dataset']['data_folder'],
                statistics=self.config['dataset']['statistics'],
                encoder=enc.encode,
                perform_FFT='ADC'
            )

        # Split dataset into training and validation
        self.train_loader, self.val_loader, _ = CreateDataLoaders(
            dataset, 
            batch_size=4,  # You can parametrize batch_size as needed
            config=self.config['dataloader'], 
            seed=self.config['seed']
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
    

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
    #device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    # Set up the config as per trial parameters
    optuna_para_config = {
        "optimizer": {
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True), # original: 1-e4
            "step_size": trial.suggest_int('step_size', 5, 15, step=5) # original: 10
        },
        #  "model": {
        #       "embed_dim": trial.suggest_categorical('embed_dim', [24, 48]), 
        # #     "mimo_layer": trial.suggest_int('mimo_layer', 64, 192, step=64) # original: 192
        #  },
        #"batch_size": trial.suggest_categorical('batch_size', [4, 8]), # original: 4
        #"threshold":  trial.suggest_float("FFT_confidence_threshold", 0.1, 0.2, step=0.05) # original: 0.2
    }


    # Initialize your Lightning DataModule
    data_module = DataModule(config)

    # set num_epochs
    # num_epochs = int(config['num_epochs'])
    #batch_size = config['dataloader']['train']['batch_size'] # 
    # batch_size = 4 #optuna_para_config['batch_size']
    # if batch_size == 4 or batch_size == 8:
    #     num_epochs = 150
    # elif batch_size == 16 or batch_size == 32:
    #     num_epochs = 200
    
    
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'mAP': [], 'mAR': [], 'val_f1': [], 'train_f1': []}

    enc = ra_encoder(geometry=config['dataset']['geometry'],
                         statistics=config['dataset']['statistics'],
                         regression_layer=2)
    
    config_optuna = dict(trial.params)
    config_optuna["trial.number"] = trial.number

    # Create a LightningModule
    model = LightningNet(config=config, step_size=optuna_para_config["optimizer"]["step_size"], 
                         lr=optuna_para_config["optimizer"]["lr"], history=history
                         , encoder=enc, data_module=data_module, config_optuna = config_optuna)

   


    # Logger setup (optional)
    # Check if the current process is the main process (rank 0)
    # print("checking local rank: ", os.environ.get('LOCAL_RANK'))
    # if os.environ.get('LOCAL_RANK') == '0':
    #     print(".......os.environ.get('LOCAL_RANK') == '0' .....")
    #     # Initialize W&B
    #     wandb.init(
    #     project=config['optuna_project'],
    #     entity="chu06-imec",  # NOTE: this entity depends on your wandb account.
    #     config=config_optuna,
    #     group='TFFTRadNet_optimization',
    #     reinit=True,
    #     )
    #     
    # else:
    #     wandb_logger = None
    wandb_logger = WandbLogger(log_model="all",
                                #project=config['optuna_project'],
                                 #entity="chu06-imec",  # NOTE: this entity depends on your wandb account.
                                 #config=config_optuna,
                                 #group='TFFTRadNet_optimization',
                                 #reinit=True,
                                 )

    # Check if the trial is a real optuna.Trial
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="f1_score")


    # Define the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Metric to monitor
        dirpath=config['output']['dir'],  # Directory to save checkpoints
        filename=f"{exp_name}___{{epoch:02d}}-{{val_loss:.2f}}",  # Checkpoint filename
        save_top_k=-1,  # Save the best model only
        mode="min",  # "min" if lower metric is better, "max" if higher is better
    )

    # Configure the Trainer
    trainer = Trainer(
        max_epochs=config['num_epochs'],
        accelerator='gpu',
        strategy=DDPStrategy(find_unused_parameters=True),
        devices=2,  # Change as needed
        log_every_n_steps=50,
        check_val_every_n_epoch=-1,
        #logger=wandb_logger,
        callbacks=[checkpoint_callback, pruning_callback],  # Monitor F1 score
    )

    #hyperparameters = config_optuna
    #trainer.logger.log_hyperparams(hyperparameters)

    print("start trainer fitting")
    trainer.fit(model, datamodule=data_module)

    pruning_callback.check_pruned()

    return trainer.callback_metrics["f1_score"].item()

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

    # Get the server's hostname
    hostname = socket.gethostname()

    # Print the hostname
    print(f"Server hostname: {hostname}")

    if hostname == "lv5clsgpu003.imec.be":
        storage_link = "postgresql://chu06:hello_psql@localhost:6089/mydb"
    elif hostname == "lv3clscsa001.imec.be":
        storage_link = "postgresql://chu06:hello_psql@localhost:59329/mydb"
    else:
        raise Exception("no postgresql initiate")

    # Create a study
    # optuna without parallel
    study = optuna.create_study(direction='maximize', study_name='FFTRadNet_optimization', 
                                storage = storage_link,
                                load_if_exists = True,
                                pruner = optuna.pruners.MedianPruner() 
                                #pruner=optuna.pruners.PercentilePruner(50.0, n_startup_trials=5,
                                #           n_warmup_steps=30, interval_steps=10)
                                           )
    


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
