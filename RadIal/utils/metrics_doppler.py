from typing import Final
import torch
import os
import json
import numpy as np
import pkbar
import argparse
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time

import matplotlib.pyplot as plt

from torchvision.ops import box_iou

def RA_to_cartesian_box(data):
    L = 4
    W = 1.8
    
    boxes = []
    for i in range(len(data)):
        # data[i][0]: R, data[i][1]: A, data[i][2]: V,  data[i][3]: C  
        x = np.sin(np.radians(data[i][1])) * data[i][0]
        y = np.cos(np.radians(data[i][1])) * data[i][0]
        v = data[i][2]

        boxes.append([x - W/2,y,x + W/2,y, x + W/2,y+L,x - W/2,y+L,v])
              
    return boxes

def perform_nms(valid_class_predictions, valid_box_predictions, nms_threshold):

    # sort the detections such that the entry with the maximum confidence score is at the top
    #print("valid_box_predictions: ", valid_box_predictions.shape)
    #print("sorted_box_predictions: ", sorted_box_predictions.shape)
    sorted_indices = np.argsort(valid_class_predictions)[::-1]
    sorted_box_predictions = valid_box_predictions[sorted_indices]
    sorted_class_predictions = valid_class_predictions[sorted_indices]
    #print("sorted_box_predictions.shape[0]: ", sorted_box_predictions.shape[0])
    for i in range(sorted_box_predictions.shape[0]):
        # get the IOUs of all boxes with the currently most certain bounding box
        try:
            start_time = time.time()
            #print("sorted_box_predictions[i, :]: ", sorted_box_predictions[i, :].shape)
            #print("sorted_box_predictions[i + 1:, :]: ", sorted_box_predictions[i + 1:, :].shape)
            ious = np.zeros((sorted_box_predictions.shape[0]))
            ious[i + 1:] = bbox_iou(sorted_box_predictions[i, :], sorted_box_predictions[i + 1:, :])
            end_time = time.time()
            #print("computation time: ", end_time-start_time)
            #ious[i + 1:] = bbox_iou_pytorch(sorted_box_predictions[i, :], sorted_box_predictions[i + 1:, :])
        except ValueError:
            break
        except IndexError:
            break

        # eliminate all detections which have IoU > threshold
        overlap_mask = np.where(ious < nms_threshold, True, False)
        sorted_box_predictions = sorted_box_predictions[overlap_mask]
        sorted_class_predictions = sorted_class_predictions[overlap_mask]

    return sorted_class_predictions, sorted_box_predictions




def bbox_iou(box1, boxes):

    # currently inspected box
    box1 = box1.reshape((4,2))
    rect_1 = Polygon([(box1[0, 0], box1[0, 1]), (box1[1, 0], box1[1, 1]), (box1[2, 0], box1[2, 1]),
                      (box1[3, 0], box1[3, 1])])
    area_1 = rect_1.area

    # IoU of box1 with each of the boxes in "boxes"
    ious = np.zeros(boxes.shape[0])
    for box_id in range(boxes.shape[0]):
        box2 = boxes[box_id]
        box2 = box2.reshape((4,2))
        rect_2 = Polygon([(box2[0, 0], box2[0, 1]), (box2[1, 0], box2[1, 1]), (box2[2, 0], box2[2, 1]),
                          (box2[3, 0], box2[3, 1])])
        area_2 = rect_2.area

        # get intersection of both bounding boxes
        inter_area = rect_1.intersection(rect_2).area

        # compute IoU of the two bounding boxes
        iou = inter_area / (area_1 + area_2 - inter_area)

        ious[box_id] = iou
        #print("area_1: ", area_1, "area_2: ", area_2)
        # Record the end time and calculate the computation time for this iteration.
        # if inter_area > 0:
        #     print("area_1: ", area_1, "area_2: ", area_2, "inter_area: ", inter_area)

    return ious

def process_predictions_FFT(batch_predictions, confidence_threshold=0.1, nms_threshold=0.05):

    # process targets and perform NMS for each prediction in batch
    final_batch_predictions = None  # store final bounding box predictions
    # batch_predictions (i, 4) # row: # of targets; column: R A V C
    point_cloud_reg_predictions = RA_to_cartesian_box(batch_predictions) # (i, 8+1) 8:xy, 1:v
    #print("finish RA_to_cartesian_box")
    #print("batch_predictions: ", batch_predictions.shape)
    point_cloud_reg_predictions = np.asarray(point_cloud_reg_predictions)
    point_cloud_class_predictions = batch_predictions[:,-1] # (i, 1) # row: # of targets; column: R A V C [:, -1]: C

    prediction = point_cloud_class_predictions
    plt.figure(figsize=(8, 6))
    plt.hist(prediction.ravel(), bins=50, color='purple')
    plt.title(f'Histogram of Prediction Values ')
    plt.xlabel('Prediction Value')
    plt.ylabel('Frequency')
    plt.ylim(0, 1000)
    filepath = os.path.join("/imec/other/dl4ms/chu06/public/", 'histogram_doppler.png')
    plt.savefig(filepath)
    print(f'Plot saved to {filepath}')
    plt.close()
    raise Exception("finishing plot hist")
    # get valid detections
    validity_mask = np.where(point_cloud_class_predictions > confidence_threshold, True, False)
    #print("confidence_threshold: ", confidence_threshold)
    
    valid_box_predictions = point_cloud_reg_predictions[validity_mask]
    valid_class_predictions = point_cloud_class_predictions[validity_mask]
    #print("point_cloud_reg_predictions: ", point_cloud_reg_predictions.shape)
    #print("valid_box_predictions: ", valid_box_predictions.shape)

    
    # perform Non-Maximum Suppression
    #print("perform_nms")
    final_class_predictions, final_box_predictions = perform_nms(valid_class_predictions, valid_box_predictions,
                                                                 nms_threshold)

    
    # concatenate point_cloud_id, confidence score and bounding box prediction | shape: [N_FINAL, 1+1+8]
    final_Object_predictions = np.hstack((final_class_predictions[:, np.newaxis],
                                               final_box_predictions))


    return final_Object_predictions

def GetFullMetrics(predictions,object_labels,range_min=5,range_max=100,IOU_threshold=0.5):
    perfs = {}
    precision = []
    recall = []
    iou_threshold = []
    RangeError = []
    AngleError = []
    VelError = []

    out = []

    for threshold in np.arange(0.1,0.96,0.1):

        iou_threshold.append(threshold)

        TP = 0
        FP = 0
        FN = 0
        NbDet = 0
        NbGT = 0
        NBFrame = 0
        range_error=0
        angle_error=0
        vel_error=0
        nbObjects = 0

        for frame_id in range(len(predictions)):
            #if frame_id % 100 == 0:
            #    print(frame_id)

            pred= predictions[frame_id] # [i, 4]  encoder.decode R A V C
            labels = object_labels[frame_id] # [4, 4] data[2] pd.read_csv(csv_file).to_numpy()
            labels = labels[:, -1]

            # get final bounding box predictions
            Object_predictions = []
            ground_truth_box_corners = []

            if(len(pred)>0):
                # process_predictions_FFT:  R A V C -> C  R A V 
                Object_predictions = process_predictions_FFT(pred,confidence_threshold=threshold) # [0, :] cls, [1:, :] reg

            if(len(Object_predictions)>0):
                max_distance_predictions = (Object_predictions[:,2]+Object_predictions[:,4])/2
                ids = np.where((max_distance_predictions>=range_min) & (max_distance_predictions <= range_max) )
                Object_predictions = Object_predictions[ids]

            NbDet += len(Object_predictions)

            if(len(labels)>0):
                ids = np.where((labels[:,0]>=range_min) & (labels[:,0] <= range_max))
                labels = labels[ids]

            if(len(labels)>0):
                ground_truth_box_corners = np.asarray(RA_to_cartesian_box(labels)) # (3, 8+1) 8:xy, 1:v
                NbGT += ground_truth_box_corners.shape[0] # 3

            # valid predictions and labels exist for the currently inspected point cloud
            if len(ground_truth_box_corners)>0 and len(Object_predictions)>0:

                used_gt = np.zeros(len(ground_truth_box_corners))
                for pid, prediction in enumerate(Object_predictions):
                    #print("prediction: ", prediction.shape, "ground_truth_box_corners: ", ground_truth_box_corners.shape)
                    iou = bbox_iou(prediction[1:9], ground_truth_box_corners[:, 0:8])
                    ids = np.where(iou>=IOU_threshold)[0]


                    if(len(ids)>0):
                        TP += 1
                        used_gt[ids]=1

                        # cummulate errors
                        range_error += np.sum(np.abs(ground_truth_box_corners[ids,-3] - prediction[-3]))
                        angle_error += np.sum(np.abs(ground_truth_box_corners[ids,-2] - prediction[-2]))
                        vel_error += np.sum(np.abs(ground_truth_box_corners[ids,-1] - prediction[-1]))
                        nbObjects+=len(ids)
                    else:
                        FP+=1
                FN += np.sum(used_gt==0)


            elif(len(ground_truth_box_corners)==0):
                FP += len(Object_predictions)
            elif(len(Object_predictions)==0):
                FN += len(ground_truth_box_corners)



        if(TP!=0):
            precision.append( TP / (TP+FP)) # When there is a detection, how much I m sure
            recall.append(TP / (TP+FN))
        else:
            precision.append( 0) # When there is a detection, how much I m sure
            recall.append(0)

        RangeError.append(range_error/nbObjects)
        AngleError.append(angle_error/nbObjects)
        VelError.append(vel_error/nbObjects)

    perfs['precision']=precision
    perfs['recall']=recall

    F1_score = (np.mean(precision)*np.mean(recall))/((np.mean(precision) + np.mean(recall))/2)


    output_file = os.path.join('TFFTRadNet_detection_score_ADC_vel.txt')
    print("Saving scores to:", output_file)
    with open(output_file, 'a') as f:
        f.write('------- Detection Scores - IOU Threshold {0} ------------\n'.format(IOU_threshold))
        f.write('  mAP: {0}\n'.format(np.mean(perfs['precision'])))
        f.write('  mAR: {0}\n'.format(np.mean(perfs['recall'])))
        f.write('  F1 score: {0}\n'.format(F1_score))
        f.write('------- Regression Errors------------\n')
        f.write('  Range Error:: {0}\n'.format(np.mean(RangeError)))
        f.write('  Angle Error: {0}\n'.format(np.mean(AngleError)))  
        f.write('  Vel Error: {0}\n'.format(np.mean(VelError)))      

    # print('------- Detection Scores - IOU Threshold {0} ------------'.format(IOU_threshold))
    # print('  mAP:',np.mean(perfs['precision']))
    # print('  mAR:',np.mean(perfs['recall']))
    # print('  F1 score:',F1_score)

    # print('------- Regression Errors------------')
    # print('  Range Error:',np.mean(RangeError),'m')
    # print('  Angle Error:',np.mean(AngleError),'degree')
    # #print('  Range Error: N/A')
    # #print('  Angle Error: N/A')
    return [IOU_threshold,np.mean(perfs['precision']),np.mean(perfs['recall']),F1_score,np.mean(RangeError),np.mean(AngleError)]




def GetDetMetrics(predictions,object_labels,threshold=0.2,range_min=5,range_max=70,IOU_threshold=0.2):

    TP = 0
    FP = 0
    FN = 0
    NbDet=0
    NbGT=0
   
    # get final bounding box predictions
    Object_predictions = []
    ground_truth_box_corners = []    
    labels=[]       

    if(len(predictions)>0):
        #print("process_predictions_FFT")
        Object_predictions = process_predictions_FFT(predictions,confidence_threshold=threshold)

        #raise Exception("error")

    if(len(Object_predictions)>0):
        max_distance_predictions = (Object_predictions[:,2]+Object_predictions[:,4])/2
        ids = np.where((max_distance_predictions>=range_min) & (max_distance_predictions <= range_max) )
        Object_predictions = Object_predictions[ids]

    NbDet = len(Object_predictions)
 
    if(len(object_labels)>0):
        ids = np.where((object_labels[:,0]>=range_min) & (object_labels[:,0] <= range_max))
        labels = object_labels[ids]
    if(len(labels)>0):
        #print("RA_to_cartesian_box")
        ground_truth_box_corners = np.asarray(RA_to_cartesian_box(labels))
        NbGT = len(ground_truth_box_corners)

    # valid predictions and labels exist for the currently inspected point cloud
    if NbDet>0 and NbGT>0:

        used_gt = np.zeros(len(ground_truth_box_corners))
        
        #start_time = time.time()
        #print("GetDetMetrics: ", start_time)
        #print("Object_predictions: ", Object_predictions.shape)
        for pid, prediction in enumerate(Object_predictions):

            iou = bbox_iou(prediction[1:9], ground_truth_box_corners[:, 0:8])
            #iou = bbox_iou_pytorch(prediction[1:], ground_truth_box_corners)
            ids = np.where(iou>=IOU_threshold)[0]

            if(len(ids)>0):
                TP += 1
                used_gt[ids]=1
            else:
                FP+=1
        FN += np.sum(used_gt==0)
        #end_time = time.time()
        #computation_time = end_time - start_time
        #print(f"Computation time = {computation_time:.4f} seconds")

    elif(NbGT==0):
        FP += NbDet
    elif(NbDet==0):
        FN += NbGT
        
    #print("TP: ", TP, "FP: ", FP, "FN: ", FN)
    return TP,FP,FN


def GetSegMetrics(PredMap,label_map):

    # Segmentation
    pred = PredMap.reshape(-1)>=0.5
    label = label_map.reshape(-1)

    intersection = np.abs(pred*label).sum()
    union = np.sum(label) + np.sum(pred) -intersection
    iou = intersection /union

    return iou

class Metrics():
    def __init__(self,):
        
        self.iou = []
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.precision = 0
        self.recall = 0
        self.mIoU =0

    def update(self,PredMap,label_map,ObjectPred,Objectlabels,threshold=0.2,range_min=5,range_max=70):
        #print("cpu")
        # metrics.update(pred_map,true_map,np.asarray(encoder.decode(pred_obj,0.05)),true_obj,
        #                   threshold=0.2,range_min=0,range_max=85.3) 
        # encoder.decode  coordinates.append([R,A,V,C])

        #TP,FP,FN = GetDetMetrics(ObjectPred,Objectlabels,threshold=0.2,range_min=range_min,range_max=range_max)
        TP,FP,FN = GetDetMetrics(ObjectPred,Objectlabels,threshold,range_min=range_min,range_max=range_max)

        self.TP += TP
        self.FP += FP
        self.FN += FN

    def reset(self,):
        self.iou = []
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.precision = 0
        self.mIoU =0

    def GetMetrics(self,):
        
        if(self.TP+self.FP!=0):
            self.precision = self.TP / (self.TP+self.FP)
        if(self.TP+self.FN!=0):
            self.recall = self.TP / (self.TP+self.FN)

        if(len(self.iou)>0):
            self.mIoU = np.asarray(self.iou).mean()

        return self.precision,self.recall,self.mIoU , self.TP, self.FP, self.FN 


