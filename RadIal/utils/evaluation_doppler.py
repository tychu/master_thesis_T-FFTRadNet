import torch
import numpy as np
from .metrics_doppler import GetFullMetrics, Metrics
import pkbar

import matplotlib.pyplot as plt
import os
import torch.nn as nn
from utils.plots import matrix_plot, detection_plot
import time

def run_evaluation(net, loader,encoder, check_perf=False, detection_loss=None,losses_params=None, optuna_config=None, config=None):

    metrics = Metrics()
    metrics.reset()

    net.eval()
    running_loss = 0.0
    

    #encoder_threshold = trial.suggest_float("encoder_threshold", 0.01, 0.07, step=0.02) # paper set 0.05
    #FFT_confidence_threshold = trial.suggest_float("FFT_confidence_threshold", 0.1, 0.5, step=0.1) # paper set 0.2


    for i, data in enumerate(loader):

        # input, out_label,segmap,labels
        if config['data_mode'] == 'ADC':
            inputs = data[0].to('cuda').type(torch.complex64)

        else:
            inputs = data[0].to('cuda').float()
        #inputs = data[0].to(gpu).float()

        label_map = data[1].to('cuda').float()
        #print("eval label_map: ", label_map.shape)
        # batch_labels[:,0]:  torch.Size([4, 128, 224, 32])


        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        #if(detection_loss!=None and segmentation_loss!=None):
        if(detection_loss!=None):
            classif_loss,reg_loss = detection_loss(outputs, label_map,losses_params)           
                
            classif_loss *= losses_params['weight'][0]
            reg_loss *= losses_params['weight'][1]

            loss = classif_loss + reg_loss 
            # statistics
            running_loss += loss.item() * inputs.size(0)

        if(check_perf):
            out_obj = outputs.detach().cpu().numpy().copy()
            labels = data[2] # data[2]=labels(=box_labels)


            for pred_obj,true_obj in zip(out_obj,labels):
                pred_map = 0 # not used in update() for computing pred_map=PredMap
                true_map = 0 # not used in update() for computing ture_map=lable_map

                #metrics.update(pred_map[0],true_map,np.asarray(encoder.decode(pred_obj,0.05)),true_obj,
                #            threshold=0.2,range_min=5,range_max=100) 
                #start_time = time.time()
                #print("cpu: ", start_time)
                #print("pred_obj: ", pred_obj.shape) #(4, 128, 224, 32)
                metrics.update(pred_map,true_map,np.asarray(encoder.decode(pred_obj,0.05)),true_obj,
                           threshold=0.2,range_min=0,range_max=85.3) 
                #end_time = time.time()
                #computation_time = end_time - start_time
                #print(f"Computation time = {computation_time:.4f} seconds")
                # metrics.update(pred_map,true_map,np.asarray(encoder.decode(pred_obj,optuna_config['threshold'])),true_obj,
                #              threshold=optuna_config['threshold'],range_min=0,range_max=345) 
                




        

    mAP,mAR, mIoU, TP, FP, FN = metrics.GetMetrics()

    return {'loss':running_loss, 'mAP':mAP, 'mAR':mAR, 'mIoU':mIoU, 'TP': TP, 'FP':FP, 'FN':FN}
    #return {'loss':running_loss}

def run_evaluation_lightning(net, loader,encoder, check_perf=False, detection_loss=None,losses_params=None, optuna_config=None, config=None):

    metrics = Metrics()
    metrics.reset()

    net.eval()
    running_loss = 0.0
    
    for i, data in enumerate(loader):

        # input, out_label,segmap,labels
        if config['data_mode'] == 'ADC':
            inputs = data[0].type(torch.complex64)

        else:
            inputs = data[0].float()
        #inputs = data[0].to(gpu).float()

        label_map = data[1].float()


        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        #if(detection_loss!=None and segmentation_loss!=None):
        if(detection_loss!=None):
            classif_loss,reg_loss = detection_loss(outputs, label_map,losses_params)           
                
            classif_loss *= losses_params['weight'][0]
            reg_loss *= losses_params['weight'][1]

            loss = classif_loss + reg_loss 
            # statistics
            running_loss += loss.item() * inputs.size(0)

        if(check_perf):
            out_obj = outputs.detach().cpu().numpy().copy()
            labels = data[2] # data[2]=labels(=box_labels)


            for pred_obj,true_obj in zip(out_obj,labels):
                pred_map = 0 # not used in update() for computing pred_map=PredMap
                true_map = 0 # not used in update() for computing ture_map=lable_map

                metrics.update(pred_map,true_map,np.asarray(encoder.decode(pred_obj,0.05)),true_obj,
                           threshold=0.2,range_min=0,range_max=345) 


    mAP,mAR, mIoU, TP, FP, FN = metrics.GetMetrics()

    return {'loss':running_loss, 'mAP':mAP, 'mAR':mAR, 'mIoU':mIoU, 'TP': TP, 'FP':FP, 'FN':FN}
    #return {'loss':running_loss}

def run_evaluation_ddp(net, loader,encoder, gpu, check_perf=False, detection_loss=None,losses_params=None, optuna_config=None, config=None):

    metrics = Metrics()
    metrics.reset()

    net.eval()
    running_loss = 0.0
    

    #encoder_threshold = trial.suggest_float("encoder_threshold", 0.01, 0.07, step=0.02) # paper set 0.05
    #FFT_confidence_threshold = trial.suggest_float("FFT_confidence_threshold", 0.1, 0.5, step=0.1) # paper set 0.2


    for i, data in enumerate(loader):

        # input, out_label,segmap,labels
        if config['data_mode'] == 'ADC':
            inputs = data[0].to(gpu).type(torch.complex64)

        else:
            inputs = data[0].to(gpu).float()
        #inputs = data[0].to(gpu).float()
        
        label_map = data[1].to(gpu).float()


        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        #if(detection_loss!=None and segmentation_loss!=None):
        if(detection_loss!=None):
            classif_loss,reg_loss = detection_loss(outputs, label_map,losses_params)           
                
            classif_loss *= losses_params['weight'][0]
            reg_loss *= losses_params['weight'][1]

            loss = classif_loss + reg_loss 
            # statistics
            running_loss += loss.item() * inputs.size(0)

        if(check_perf):
            out_obj = outputs.detach().cpu().numpy().copy()
            labels = data[2] # data[2]=labels(=box_labels)


            for pred_obj,true_obj in zip(out_obj,labels):
                pred_map = 0 # not used in update() for computing pred_map=PredMap
                true_map = 0 # not used in update() for computing ture_map=lable_map

                #metrics.update(pred_map[0],true_map,np.asarray(encoder.decode(pred_obj,0.05)),true_obj,
                #            threshold=0.2,range_min=5,range_max=100) 
                metrics.update(pred_map,true_map,np.asarray(encoder.decode(pred_obj,0.05)),true_obj,
                           threshold=0.2,range_min=0,range_max=345) 
                # metrics.update(pred_map,true_map,np.asarray(encoder.decode(pred_obj,optuna_config['threshold'])),true_obj,
                #              threshold=optuna_config['threshold'],range_min=0,range_max=345) 
                




        

    mAP,mAR, mIoU, TP, FP, FN = metrics.GetMetrics()

    return {'loss':running_loss, 'mAP':mAP, 'mAR':mAR, 'mIoU':mIoU, 'TP': TP, 'FP':FP, 'FN':FN}
    #return {'loss':running_loss}

def run_FullEvaluation(net,loader,encoder,iou_threshold=0.5,config=None):

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

        out_obj = outputs['Detection'].detach().cpu().numpy().copy()
        out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()

        labels_object = data[3]
        label_freespace = data[2].numpy().copy()

        for pred_obj,pred_map,true_obj,true_map in zip(out_obj,out_seg,labels_object,label_freespace):

            predictions['prediction']['objects'].append( np.asarray(encoder.decode(pred_obj,0.05)))
            predictions['label']['objects'].append(true_obj)

            predictions['prediction']['freespace'].append(pred_map[0])
            predictions['label']['freespace'].append(true_map)


        kbar.update(i)

    iou_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for iou_ in iou_list:
        results.append(GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=5,range_max=100,IOU_threshold=iou_))

    # mIoU = []
    # for i in range(len(predictions['prediction']['freespace'])):
    #     # 0 to 124 means 0 to 50m
    #     pred = predictions['prediction']['freespace'][i][:124].reshape(-1)>=0.5
    #     label = predictions['label']['freespace'][i][:124].reshape(-1)

    #     intersection = np.abs(pred*label).sum()
    #     union = np.sum(label) + np.sum(pred) -intersection
    #     iou = intersection /union
    #     mIoU.append(iou)


    # mIoU = np.asarray(mIoU).mean()
    # print('------- Freespace Scores ------------')
    # print('  mIoU',mIoU*100,'%')

def run_FullEvaluation_(net,loader,encoder,iou_threshold=0.5,config=None):

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

        out_obj = outputs.detach().cpu().numpy().copy()

        labels_object = data[2]


        for pred_obj,true_obj in zip(out_obj,labels_object):

            predictions['prediction']['objects'].append( np.asarray(encoder.decode(pred_obj,0.05)))
            predictions['label']['objects'].append(true_obj)


        kbar.update(i)

    iou_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for iou_ in iou_list:
        #results.append(GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=5,range_max=100,IOU_threshold=iou_))
        results.append(GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=0,range_max=345,IOU_threshold=iou_))
    print(results)

    return results


def run_iEvaluation(net,loader,encoder,epochs, datamode,iou_threshold=0.5):

    net.eval()
    model_mode = net.training


    
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    print('Generating Predictions...')
    predictions = {'prediction':{'objects':[],'freespace':[]},'label':{'objects':[],'freespace':[]}}
    for i, data in enumerate(loader):

        if (i % 5 == 0): #len(loader) - 1:
                    # input, out_label,segmap,labels
            inputs = data[0].to('cuda').float()


            with torch.set_grad_enabled(False):
                outputs = net(inputs)

            label_map = data[1].to('cuda').float() # debugging

            out_obj = outputs['Detection'].detach().cpu().numpy().copy()
            out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
            
            matrix_plot(outputs['Detection'], label_map, model_mode, datamode, epoch=epochs, batch = i) # debugging  
            detection_plot(outputs['Detection'], label_map, model_mode, datamode, epoch=epochs, batch = i)   

            labels_object = data[3]
            label_freespace = data[2].numpy().copy()
            print("out_obj : ", out_obj.shape)
                
            for pred_obj,pred_map,true_obj,true_map in zip(out_obj,out_seg,labels_object,label_freespace):
                
                print("pred_obj : ", pred_obj.shape)
                print("true_obj : ", true_obj.shape)
                print("np.asarray(encoder.decode(pred_obj,0.05)) : ", np.asarray(encoder.decode(pred_obj,0.05)).shape)
                predictions['prediction']['objects'].append( np.asarray(encoder.decode(pred_obj,0.05)))
                predictions['label']['objects'].append(true_obj)

                predictions['prediction']['freespace'].append(pred_map[0])
                predictions['label']['freespace'].append(true_map)
            
            print("predictions['prediction']['objects'] : ", len(predictions['prediction']['objects']))
            print("predictions['label']['objects'] : ", len(predictions['label']['objects']))
            #GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=0,range_max=345,IOU_threshold=0.5)

        kbar.update(i)

