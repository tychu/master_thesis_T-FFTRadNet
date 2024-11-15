import torch
import torch.nn.functional as F
import torch.nn as nn

import torch.nn as nn
class FocalLoss(nn.Module):
    """
    Focal loss class. Stabilize training by reducing the weight of easily classified background sample and focussing
    on difficult foreground detections.
    """

    def __init__(self, gamma=0, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, prediction, target):

        # get class probability
        pt = torch.where(target == 1.0, prediction, 1-prediction)
        # pt (batch, range_bin, angle_bin, doppler) 

        # compute focal loss
        loss = -1 * (1-pt)**self.gamma * torch.log(pt+1e-6)
        # loss (batch, range_bin, angle_bin, doppler) 

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum() # single scalar value
    
def pixor_loss(batch_predictions, batch_labels,param):


    #########################
    #  classification loss  #
    #########################
    classification_prediction = batch_predictions[:, 0, :, :, :].contiguous().flatten()
    # batch_predictions = output (batch, output_dim, doppler, range_bin, angle_bin) 
    # [:, 0, :, :]: classification, [:, 1, :, :]: range,  [:, 2, :, :]: angle
    classification_label = batch_labels[:, 0, :, :, :].contiguous().flatten()
    print("classification_prediction: ", classification_prediction.shape, "batch_predictions: ", batch_predictions.shape)
    print("classification_label: ", classification_label.shape, "batch_labels: ", batch_labels.shape)
    if(param['classification']=='FocalLoss'):
        focal_loss = FocalLoss(gamma=2)
        classification_loss = focal_loss(classification_prediction, classification_label)
        
    else:
        classification_loss = F.binary_cross_entropy(classification_prediction.double(), classification_label.double(),reduction='sum')

    
    #####################
    #  Regression loss  #
    #####################

    T = batch_labels[:,1:]
    P = batch_predictions[:,1:] # only take range and angle, exclude classificstion = batch_predictions[:, 0]
    M = batch_labels[:,0].unsqueeze(1)
    #print("batch_labels[:,0]: ", batch_labels[:,0].shape)
    #print("p: ", P.shape, "M: ", M.shape)

    if(param['regression']=='SmoothL1Loss'):
        reg_loss_fct = nn.SmoothL1Loss(reduction='sum')
    else:
        reg_loss_fct = nn.L1Loss(reduction='sum')
    
    regression_loss = reg_loss_fct(P*M,T)
    NbPts = M.sum()
    if(NbPts>0):
        regression_loss/=NbPts

    return classification_loss,regression_loss