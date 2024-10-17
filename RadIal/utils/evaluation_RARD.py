import torch
import numpy as np
from .metrics_RARD import GetFullMetrics, Metrics
import pkbar

import matplotlib.pyplot as plt
import os
import torch.nn as nn
from utils.plots import matrix_plot, detection_plot
import time
from loss.loss_RARD import pixor_loss, MVloss

def run_evaluation(net, loader,encoder, check_perf=False, detection_loss=None,losses_params=None, optuna_config=None, config=None):

    metrics = Metrics()
    metrics.reset()

    net.eval()
    running_loss = 0.0
    running_mv_loss = 0.0
    running_cls_loss = 0.0
    running_reg_loss = 0.0
    

    #encoder_threshold = trial.suggest_float("encoder_threshold", 0.01, 0.07, step=0.02) # paper set 0.05
    #FFT_confidence_threshold = trial.suggest_float("FFT_confidence_threshold", 0.1, 0.5, step=0.1) # paper set 0.2


    for i, data in enumerate(loader):

        # input, out_label,segmap,labels
        if config['data_mode'] == 'ADC':
            inputs = data[0].to('cuda').type(torch.complex64)

        else:
            inputs = data[0].to('cuda').float()
        #inputs = data[0].to(gpu).float()

        #label_map = data[1].to('cuda').float()
        label_ra_map = data[1].to('cuda').float() # matlab_dataset.py: out_label_ra
        label_rd_map = data[2].to('cuda').float() # matlab_dataset.py: out_label_rd


        with torch.set_grad_enabled(False):
            outputs = net(inputs) # outputs['RA']  outputs['RD']

        #if(detection_loss!=None and segmentation_loss!=None):
        if(detection_loss!=None):
            #print("outputs['RA']: ", outputs['RA'].shape, "outputs['RD']: ", outputs['RD'].shape)
            classif_loss_ra, reg_loss_ra = detection_loss(outputs['RA'], label_ra_map,losses_params)    # pixor_loss
            classif_loss_rd, reg_loss_rd = detection_loss(outputs['RD'], label_rd_map,losses_params)  

            classif_loss = classif_loss_ra + classif_loss_rd
            reg_loss = reg_loss_ra + reg_loss_rd      
            mvloss = MVloss(outputs)

            ### checking loss without weight
            running_mv_loss += mvloss.item() * inputs.size(0)
            running_cls_loss += classif_loss.item() * inputs.size(0)
            running_reg_loss += reg_loss.item() * inputs.size(0)      
            ####      
                
            classif_loss *= losses_params['weight'][0]
            reg_loss *= losses_params['weight'][1]
            mvloss *= losses_params['weight'][2]

            loss = classif_loss + reg_loss + mvloss
            # statistics
            running_loss += loss.item() * inputs.size(0)

        if(check_perf):
            out_obj = outputs['RA'].detach().cpu().numpy().copy()
            labels = data[3] # data[2]=labels(=box_labels)
            #print("labels: ", labels.shape)

            labels_RD_map = data[2].detach().cpu().numpy().copy()
            #print("labels_RD_map: ", labels_RD_map.shape)
            out_RD = outputs['RD'].detach().cpu().numpy().copy()


            for pred_RA,true_obj, true_RD_map, pre_RD  in zip(out_obj,labels, labels_RD_map, out_RD):
                pred_map = 0 # not used in update() for computing pred_map=PredMap
                true_map = 0 # not used in update() for computing ture_map=lable_map

                metrics.update_RD(true_RD_map, pre_RD, threshold=0.2)
                #raise Exception("checking update_RD")
                metrics.update(pred_map,true_map,np.asarray(encoder.decode(pred_RA,pre_RD,0.05)),true_obj,
                           threshold=0.2,range_min=0,range_max=85.3) 
                
    mAP,mAR, mIoU, TP, FP, FN = metrics.GetMetrics()
    miou_RD = metrics.GetMetrics_RD()

    return {'loss':running_loss,"running_mv_loss":running_mv_loss,"running_cls_loss":running_cls_loss,"running_reg_loss": running_reg_loss,'mAP':mAP, 'mAR':mAR, 'mIoU':mIoU, 'TP': TP, 'FP':FP, 'FN':FN,"miou_RD":miou_RD}
    #return {'loss':running_loss}



def run_FullEvaluation_(net,loader,encoder,iou_threshold=0.5,config=None):

    metrics = Metrics()
    metrics.reset()

    net.eval()
    results = []
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    print('Generating Predictions...')
    predictions = {'prediction':{'objects':[],'freespace':[]},'label':{'objects':[],'freespace':[]}}
    for i, data in enumerate(loader):
        # input, out_label,segmap,labels
        if config['data_mode'] == 'ADC':
            inputs = data[0].to('cuda').type(torch.complex64)

        else:
            inputs = data[0].to('cuda').float()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        #out_obj = outputs.detach().cpu().numpy().copy()

        labels_object =  data[3]
        out_obj = outputs['RA'].detach().cpu().numpy().copy()
        #print("labels_object: ", labels_object.shape)

        labels_RD_map = data[2].detach().cpu().numpy().copy()
        labels_RA_map = data[1].detach().cpu().numpy().copy()
        #print("labels_RD_map: ", labels_RD_map.shape)
        out_RD = outputs['RD'].detach().cpu().numpy().copy()


        for pred_RA,true_obj, true_RD_map, true_RA_map, pre_RD  in zip(out_obj,labels_object, labels_RD_map, labels_RA_map, out_RD):

            predictions['prediction']['objects'].append( np.asarray(encoder.decode(pred_RA,pre_RD,0.05)))
            predictions['label']['objects'].append(true_obj)
            metrics.update_RD(true_RD_map, pre_RD, threshold=0.2)
            metrics.update_RA(true_RA_map, pred_RA, threshold=0.2)


        kbar.update(i)

    iou_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for iou_ in iou_list:
        #results.append(GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=5,range_max=100,IOU_threshold=iou_))
        results.append(GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=0,range_max=85.3,IOU_threshold=iou_))
    #print(results)
    miou_RD, prec_RD, re_RD, f1_RD = metrics.GetMetrics_RD()
    miou_RA, prec_RA, re_RA, f1_RA = metrics.GetMetrics_RA()

    return results, miou_RD, prec_RD, re_RD, f1_RD, miou_RA, prec_RA, re_RA, f1_RA


def run_iEvaluation(dir, net,loader, output_map, datamode,config=None, iou_threshold=0.5):

    net.eval()

    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    print('Generating Predictions...')
    for i, data in enumerate(loader):

        if (i % 5 == 0): #len(loader) - 1:
                    # input, out_label,segmap,labels
            if config['data_mode'] == 'ADC':
                inputs = data[0].to('cuda').type(torch.complex64)

            else:
                inputs = data[0].to('cuda').float()


            with torch.set_grad_enabled(False):
                outputs = net(inputs)

            
            print("detach")
            if output_map == "RA":
                out_obj = outputs['RA'].detach().cpu().numpy().copy()
                label_map = data[1].detach().cpu().numpy().copy() # encoded_label_ra
            elif output_map == "RD":
                out_obj = outputs['RD'].detach().cpu().numpy().copy()
                label_map = data[2].detach().cpu().numpy().copy() # encoded_label_ra
            else:
                print("output not exist")

            #print("labels_object: ", labels_object.shape)

            #labels_RD_map = data[2].detach().cpu().numpy().copy()
            #print("labels_RD_map: ", labels_RD_map.shape)
            #out_RD = outputs['RD'].detach().cpu().numpy().copy()

            print("plotting")       
            matrix_plot(dir, out_obj, label_map, output_map, datamode, i) 
            detection_plot(dir, out_obj, label_map, output_map, datamode, i)   



        kbar.update(i)

