from typing import Final
import torch
import os
import json
import numpy as np
import pkbar
import argparse
from shapely.geometry import Polygon
from shapely.ops import unary_union

from torchvision.ops import box_iou
import time

def RA_to_cartesian_box(data):
    L = 4
    W = 1.8

    #data_tensor = torch.tensor(data, dtype=torch.float32, device='cuda')
    angles_rad = torch.deg2rad(data[:, 1])
    #print("angles_rad: ", angles_rad)
    # Compute x and y using PyTorch's sin and cos, already on the GPU
    x = torch.sin(angles_rad) * data[:, 0]
    y = torch.cos(angles_rad) * data[:, 0]

    boxes_tensor = torch.stack((x - W/2, y, x + W/2, y, x + W/2, y + L, x - W/2, y + L), dim=1)
    #print("boxes_tensor: ", boxes_tensor)

    # # Compute boxes as a list of lists
    # boxes = []
    # for i in range(len(data)):
    #     x = np.sin(np.radians(data[i][1])) * data[i][0]
    #     y = np.cos(np.radians(data[i][1])) * data[i][0]
    #     print("degree: ", data[i][1])
    #     print("radians: ", np.radians(data[i][1]))
    #     boxes.append([x - W/2, y, x + W/2, y, x + W/2, y + L, x - W/2, y + L])
    #     print("boxes: ", boxes)
    #     if i > 10:
    #         raise Exception("checking ")
    # # Convert list of boxes to a NumPy array
    # boxes_np = np.array(boxes)

    # # Convert NumPy array to PyTorch tensor
    # boxes_tensor = torch.tensor(boxes_np, dtype=torch.float32, device='cuda')

    #print("RA_to_cartesian_box: ", boxes_tensor)
              
    return boxes_tensor

def perform_nms(valid_class_predictions, valid_box_predictions, nms_threshold):

    # Ensure tensors are on GPU
    #valid_class_predictions = valid_class_predictions.to('cuda')
    #valid_box_predictions = valid_box_predictions.to('cuda')

    sorted_indices = torch.argsort(valid_class_predictions, descending=True)
    sorted_box_predictions = valid_box_predictions[sorted_indices]
    sorted_class_predictions = valid_class_predictions[sorted_indices]
    print("sorted_box_predictions.shape[0]: ", sorted_box_predictions.size(0))
    #keep = []
    while sorted_box_predictions.size(0) > 0:
        largest = sorted_box_predictions[0]
        #keep.append(sorted_class_predictions[0])

        if sorted_box_predictions.size(0) == 1:
            break

        start_time = time.time()
        ious = bbox_iou(largest, sorted_box_predictions[1:])
        end_time = time.time()
        print("computation time: ", end_time-start_time)
        indices = torch.where(ious < nms_threshold)[0] + 1

        sorted_box_predictions = sorted_box_predictions[indices]
        sorted_class_predictions = sorted_class_predictions[indices]

    return sorted_class_predictions, sorted_box_predictions





def bbox_iou(box1, boxes):
# Reshape box1 and boxes into (4, 2) format
    box1 = box1.reshape((4, 2))
    boxes = boxes.reshape(-1, 4, 2)

    # Calculate the area of box1
    area_1 = 0.5 * torch.abs(
        (box1[0, 0] * (box1[1, 1] - box1[3, 1]) +
         box1[1, 0] * (box1[2, 1] - box1[0, 1]) +
         box1[2, 0] * (box1[3, 1] - box1[1, 1]) +
         box1[3, 0] * (box1[0, 1] - box1[2, 1]))
    )

    # Initialize IoUs tensor
    ious = torch.zeros(boxes.shape[0], device=box1.device)

    for box_id in range(boxes.shape[0]):
        box2 = boxes[box_id]

        # Calculate the area of box2
        area_2 = 0.5 * torch.abs(
            (box2[0, 0] * (box2[1, 1] - box2[3, 1]) +
             box2[1, 0] * (box2[2, 1] - box2[0, 1]) +
             box2[2, 0] * (box2[3, 1] - box2[1, 1]) +
             box2[3, 0] * (box2[0, 1] - box2[2, 1]))
        )

        # Compute intersection area (using a simplified box intersection, assuming axis-aligned bounding boxes)
        x_min = torch.max(box1[:, 0].min(), box2[:, 0].min())
        y_min = torch.max(box1[:, 1].min(), box2[:, 1].min())
        x_max = torch.min(box1[:, 0].max(), box2[:, 0].max())
        y_max = torch.min(box1[:, 1].max(), box2[:, 1].max())

        # Calculate the width and height of the intersection box
        inter_width = torch.clamp(x_max - x_min, min=0)
        inter_height = torch.clamp(y_max - y_min, min=0)

        # Calculate the intersection area
        inter_area = inter_width * inter_height

        # Compute IoU
        iou = inter_area / (area_1 + area_2 - inter_area)
        ious[box_id] = iou

        # if inter_area > 0:
        #     print("area_1: ", area_1, "area_2: ", area_2, "inter_area: ", inter_area)

    return ious

def process_predictions_FFT(batch_predictions, confidence_threshold=0.1, nms_threshold=0.05):

    #batch_predictions = torch.tensor(batch_predictions, dtype=torch.float32, device='cuda')

    point_cloud_reg_predictions = RA_to_cartesian_box(batch_predictions)
    point_cloud_class_predictions = batch_predictions[:, -1]

    validity_mask = point_cloud_class_predictions > confidence_threshold
    valid_box_predictions = point_cloud_reg_predictions[validity_mask]
    valid_class_predictions = point_cloud_class_predictions[validity_mask]

    final_class_predictions, final_box_predictions = perform_nms(valid_class_predictions, valid_box_predictions, nms_threshold)

    final_Object_predictions = torch.cat([
        final_class_predictions.unsqueeze(1),
        final_box_predictions
    ], dim=1)

    return final_Object_predictions

def GetFullMetrics(predictions,object_labels,range_min=5,range_max=100,IOU_threshold=0.5):
    perfs = {}
    precision = []
    recall = []
    iou_threshold = []
    RangeError = []
    AngleError = []

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
        nbObjects = 0



        for frame_id in range(len(predictions)): # len(predictions)=4
            #print("frame_id : ", frame_id)

            pred= predictions[frame_id]
            labels = object_labels[frame_id]

            # get final bounding box predictions
            Object_predictions = []
            ground_truth_box_corners = []           
            
            if(len(pred)>0):
                Object_predictions = process_predictions_FFT(pred,confidence_threshold=threshold)

            if(len(Object_predictions)>0):
                max_distance_predictions = (Object_predictions[:,2]+Object_predictions[:,4])/2
                ids = np.where((max_distance_predictions>=range_min) & (max_distance_predictions <= range_max) )
                Object_predictions = Object_predictions[ids]

            NbDet += len(Object_predictions)

            if(len(labels)>0):
                ids = np.where((labels[:,0]>=range_min) & (labels[:,0] <= range_max))
                labels = labels[ids]

            if(len(labels)>0):
                ground_truth_box_corners = np.asarray(RA_to_cartesian_box(labels))
                NbGT += ground_truth_box_corners.shape[0]

            # valid predictions and labels exist for the currently inspected point cloud
            if len(ground_truth_box_corners)>0 and len(Object_predictions)>0:

                used_gt = np.zeros(len(ground_truth_box_corners))
                for pid, prediction in enumerate(Object_predictions):
                    iou = bbox_iou(prediction[1:], ground_truth_box_corners)
                    ids = np.where(iou>=IOU_threshold)[0]

                    
                    if(len(ids)>0):
                        TP += 1
                        used_gt[ids]=1

                        # cummulate errors
                        range_error += np.sum(np.abs(ground_truth_box_corners[ids,-2] - prediction[-2]))
                        angle_error += np.sum(np.abs(ground_truth_box_corners[ids,-1] - prediction[-1]))
                        nbObjects+=len(ids)
                    else:
                        FP+=1
                FN += np.sum(used_gt==0)


            elif(len(ground_truth_box_corners)==0):
                FP += len(Object_predictions)
            elif(len(Object_predictions)==0):
                FN += len(ground_truth_box_corners)
                
            #print('TP: ', TP, 'FP: ', FP, "FN: ", FN)

        if(TP!=0):
            precision.append( TP / (TP+FP)) # When there is a detection, how much I m sure
            recall.append(TP / (TP+FN))
        else:
            precision.append( 0) # When there is a detection, how much I m sure
            recall.append(0)

        RangeError.append(range_error/nbObjects)
        AngleError.append(angle_error/nbObjects)

    mAP=np.mean(precision)
    mAR=np.mean(recall)
    print("mAP: ", mAP)
    print("precision: ", precision)
    print("mAR: ", mAR)
    print("recall: ", recall)

    F1_score = (mAP*mAR)/((mAP + mAR)/2)

    # output_file = 'detection_scores.txt'
    # print("Saving scores to:", output_file)
    # with open(output_file, 'a') as f:
    #     f.write('------- Detection Scores ------------\n')
    #     f.write('  mAP: {0}\n'.format(np.mean(perfs['precision'])))
    #     f.write('  mAR: {0}\n'.format(np.mean(perfs['recall'])))
    #     f.write('  F1 score: {0}\n'.format(F1_score))
    return mAP, mAR, F1_score, np.mean(RangeError), np.mean(AngleError)


def GetDetMetrics(predictions,object_labels,threshold=0.2,range_min=5,range_max=70,IOU_threshold=0.2, device='cuda'):

    TP = 0
    FP = 0
    FN = 0
    NbDet=0
    NbGT=0
   
    # get final bounding box predictions
    #Object_predictions = []
    #ground_truth_box_corners = []    
    #labels=[]       
    #predictions = torch.tensor(predictions, dtype=torch.float32, device='cuda')

    if predictions.size(0) > 0:
        Object_predictions = process_predictions_FFT(predictions, confidence_threshold=threshold)
    else:
        Object_predictions = torch.empty((0, 9), device=device)

    if(len(Object_predictions)>0):
        max_distance_predictions = (Object_predictions[:, 2] + Object_predictions[:, 4]) / 2
        ids = (max_distance_predictions >= range_min) & (max_distance_predictions <= range_max)
        Object_predictions = Object_predictions[ids]

    NbDet = Object_predictions.size(0)
 
    if(len(object_labels)>0):
        ids = (object_labels[:, 0] >= range_min) & (object_labels[:, 0] <= range_max)
        labels = object_labels[ids]
    else:
        labels = torch.empty((0, 9), device=device)
    if(len(labels)>0):
        ground_truth_box_corners = torch.tensor(RA_to_cartesian_box(labels))
        NbGT = ground_truth_box_corners.size(0)
    else:
        ground_truth_box_corners = torch.empty((0, 8), device=device)

    # valid predictions and labels exist for the currently inspected point cloud
    if NbDet>0 and NbGT>0:

        used_gt = torch.zeros(ground_truth_box_corners.size(0), device=device)

        for pid, prediction in enumerate(Object_predictions):
            iou = bbox_iou(prediction[1:], ground_truth_box_corners)
            ids = (iou >= IOU_threshold).nonzero(as_tuple=True)[0]

            if ids.numel() > 0:
                TP += 1
                used_gt[ids]=1
            else:
                FP+=1
        FN += (used_gt == 0).sum().item()

    elif(NbGT==0):
        FP += NbDet
    elif(NbDet==0):
        FN += NbGT
        
    print("TP: ", TP, "FP: ", FP, "FN: ", FN)
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

        # if(len(PredMap)>0):
        #     pred = PredMap.reshape(-1)>=0.5
        #     label = label_map.reshape(-1)

        #     intersection = np.abs(pred*label).sum()
        #     union = np.sum(label) + np.sum(pred) -intersection
        #     self.iou.append(intersection /union)

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

        if len(self.iou) > 0:
            self.mIoU = torch.tensor(self.iou, dtype=torch.float32, device='cuda').mean()
        else:
            self.mIoU = torch.tensor(0, dtype=torch.float32, device='cuda')

        return self.precision,self.recall,self.mIoU , self.TP, self.FP, self.FN 


