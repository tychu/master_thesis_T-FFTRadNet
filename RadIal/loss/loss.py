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
        # pt (batch, range_bin, angle_bin) 

        # compute focal loss
        loss = -1 * (1-pt)**self.gamma * torch.log(pt+1e-6)
        # loss (batch, range_bin, angle_bin) 

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum() # single scalar value
    
def pixor_loss(batch_predictions, batch_labels,param):


    #########################
    #  classification loss  #
    #########################
    classification_prediction = batch_predictions[:, 0,:, :].contiguous().flatten()
    # batch_predictions = output (batch, output_dim, range_bin, angle_bin) 
    # [:, 0, :, :]: classification, [:, 1, :, :]: range,  [:, 2, :, :]: angle
    classification_label = batch_labels[:, 0,:, :].contiguous().flatten()

    if(param['classification']=='FocalLoss'):
        focal_loss = FocalLoss(gamma=2)
        classification_loss = focal_loss(classification_prediction, classification_label)
    else:
        classification_loss = F.binary_cross_entropy(classification_prediction.double(), classification_label.double(),reduction='sum')

    
    #####################
    #  Regression loss  #
    #####################

    #regression_prediction = batch_predictions.permute([0, 2, 3, 1])[:, :, :, :-1]
    # regression_prediction(batch, range_bin, angle_bin, output_dim) 
    #print("batch_predictions.permute([0, 2, 3, 1]): ", batch_predictions.permute([0, 2, 3, 1]).shape)
    #regression_prediction = regression_prediction.contiguous().view([regression_prediction.size(0)*
    #                    regression_prediction.size(1)*regression_prediction.size(2), regression_prediction.size(3)])
    #regression_label = batch_labels.permute([0, 2, 3, 1])[:, :, :, :-1]
    #regression_label = regression_label.contiguous().view([regression_label.size(0)*regression_label.size(1)*
    #                                                       regression_label.size(2), regression_label.size(3)])

    #positive_mask = torch.nonzero(torch.sum(torch.abs(regression_label), dim=1))
    #pos_regression_label = regression_label[positive_mask.squeeze(), :]
    #pos_regression_prediction = regression_prediction[positive_mask.squeeze(), :]


    T = batch_labels[:,1:]
    P = batch_predictions[:,1:] # only take range and angle, exclude classificstion = batch_predictions[:, 0]
    M = batch_labels[:,0].unsqueeze(1)

    if(param['regression']=='SmoothL1Loss'):
        reg_loss_fct = nn.SmoothL1Loss(reduction='sum')
    else:
        reg_loss_fct = nn.L1Loss(reduction='sum')
    
    regression_loss = reg_loss_fct(P*M,T)
    NbPts = M.sum()
    if(NbPts>0):
        regression_loss/=NbPts

    return classification_loss,regression_loss