import os
import json
import argparse
import torch
import random
import numpy as np
from model.FFTRadNet_ViT_RARD_ADA import FFTRadNet_ViT
from model.FFTRadNet_ViT_RARD_ADA import FFTRadNet_ViT_ADC
from dataset.dataset import RADIal
from dataset.encoder_RARD import ra_encoder
from dataset.dataloader_RARD import CreateDataLoaders
import pkbar
import torch.nn.functional as F
from utils.evaluation_RARD import run_FullEvaluation_, run_iEvaluation
import torch.nn as nn

from dataset.matlab_dataset_RARD import MATLAB
import re

def extract_params_from_filename(self, filename):

        epoch_match = re.search(r'_epoch(\d+)', filename)

        epoch = int(epoch_match.group(1)) if epoch_match else None

        return epoch

def main(config, checkpoint, plot, eval):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    enc = ra_encoder(geometry = config['dataset']['geometry'],
                        statistics = config['dataset']['statistics'],
                        regression_layer = 2)

    if config['data_mode'] != 'ADC':
        net = FFTRadNet_ViT(patch_size = config['model']['patch_size'],
                        channels = config['model']['channels'],
                        in_chans = config['model']['in_chans'],
                        embed_dim = config['model']['embed_dim'],
                        depths = config['model']['depths'],
                        num_heads = config['model']['num_heads'],
                        drop_rates = config['model']['drop_rates'],
                        regression_layer = 2,
                        detection_head = config['model']['DetectionHead'],
                        segmentation_head = config['model']['SegmentationHead'])

        # dataset = RADIal(root_dir = config['dataset']['root_dir'],
        #                     statistics= config['dataset']['statistics'],
        #                     encoder=enc.encode,
        #                     difficult=True,perform_FFT=config['data_mode'])

        dataset = MATLAB(root_dir = config['dataset']['root_dir'],
                         folder_dir = config['dataset']['data_folder'],
                            statistics= config['dataset']['statistics'],
                            encoder=enc.encode)
    
    else:
        net = FFTRadNet_ViT_ADC(patch_size = config['model']['patch_size'],
                        channels = config['model']['channels'],
                        in_chans = config['model']['in_chans'],
                        embed_dim = config['model']['embed_dim'],
                        depths = config['model']['depths'],
                        num_heads = config['model']['num_heads'],
                        drop_rates = config['model']['drop_rates'],
                        regression_layer = 2,
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
        
    batch_size = 8
    train_loader, val_loader, test_loader = CreateDataLoaders(dataset, batch_size, config['dataloader'],config['seed'])

    print("Parameters: ",sum(p.numel() for p in net.parameters() if p.requires_grad))

    net.to('cuda')


    print('===========  Loading the model ==================:')
    dict = torch.load(checkpoint)
    net.load_state_dict(dict['net_state_dict'])

    #epochs = extract_params_from_filename(checkpoint)

    if plot:
        # Extract the directory path without the .pth file
        directory_path = os.path.dirname(checkpoint)
        run_iEvaluation(directory_path, net, train_loader, output_map = "RA", datamode='train',config=config)
        run_iEvaluation(directory_path, net, val_loader, output_map = "RA", datamode='val',config=config)
        run_iEvaluation(directory_path, net, test_loader, output_map = "RA", datamode='test',config=config)

    if eval:
        print('===========  Running the evaluation ==================:')
        output_file = os.path.join('TFFTRadNet_detection_score_ADC_RARD_ADAblock.txt')
        print("Saving scores to:", output_file)
        with open(output_file, 'a') as f:
            f.write('------- Train ------------\n')
        result_train, RD_iou_train, prec_RD_train, re_RD_train, f1_RD_train, RA_iou_train, prec_RA_train, re_RA_train, f1_RA_train = run_FullEvaluation_(net,train_loader,enc,config=config) #test_loader
        with open(output_file, 'a') as f:
            f.write('------- Validation ------------\n')
        result_val, RD_iou_val, prec_RD_val, re_RD_val, f1_RD_val, RA_iou_val, prec_RA_val, re_RA_val, f1_RA_val = run_FullEvaluation_(net,val_loader,enc,config=config) 
        with open(output_file, 'a') as f:
            f.write('------- Test ------------\n')
        result_test, RD_iou_test, prec_RD_test, re_RD_test, f1_RD_test, RA_iou_test, prec_RA_test, re_RA_test, f1_RA_test = run_FullEvaluation_(net,test_loader,enc,config=config) 

        # Transpose the data, excluding the IOU_threshold (index 0)
        transposed_train_data = list(zip(*result_train))[1:]  # Skips the first element (IOU_threshold)
        ransposed_val_data = list(zip(*result_val))[1:]  # Skips the first element (IOU_threshold)
        ransposed_test_data = list(zip(*result_test))[1:]  # Skips the first element (IOU_threshold)

        # Calculate the mean of each metric
        means_train = [np.mean(metric) for metric in transposed_train_data]
        means_val = [np.mean(metric) for metric in ransposed_val_data]
        means_test = [np.mean(metric) for metric in ransposed_test_data]
        
        with open(output_file, 'a') as f:
            f.write('------- Summary ------------\n')
            f.write('------- Train ------------\n')
            f.write('Means of Precision, Recall, F1_score, RangeError, AngleError, DopplerError:\n')
            f.write(f"{means_train}\n")
            f.write('confusion matrix of IoU of RD map, precision, recall, f1: \n')
            f.write(f"{RD_iou_train, prec_RD_train, re_RD_train, f1_RD_train}\n") 
            f.write('confusion matrix of IoU of RA map, precision, recall, f1: \n')
            f.write(f"{RA_iou_train, prec_RA_train, re_RA_train, f1_RA_train}\n")
            f.write('------- Validation ------------\n')
            f.write('Means of Precision, Recall, F1_score, RangeError, AngleError, DopplerError:\n')
            f.write(f"{means_val}\n")
            f.write('confusion matrix of IoU of RD map, precision, recall, f1: \n')
            f.write(f"{RD_iou_val, prec_RD_val, re_RD_val, f1_RD_val}\n")
            f.write('confusion matrix of IoU of RA map, precision, recall, f1: \n')
            f.write(f"{RA_iou_val, prec_RA_val, re_RA_val, f1_RA_val}\n")
            f.write('------- Test ------------\n')
            f.write('Means of Precision, Recall, F1_score, RangeError, AngleError, DopplerError:\n')
            f.write(f"{means_test}\n")
            f.write('confusion matrix of IoU of RD map, precision, recall, f1: \n')
            f.write(f"{RD_iou_test, prec_RD_test, re_RD_test, f1_RD_test}\n")
            f.write('confusion matrix of IoU of RA map, precision, recall, f1: \n')
            f.write(f"{RA_iou_test, prec_RA_test, re_RA_test, f1_RA_test}\n")




if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet Evaluation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config, args.checkpoint, args.plot, args.eval)
