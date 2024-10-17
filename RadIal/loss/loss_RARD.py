import torch
import torch.nn.functional as F
import torch.nn as nn

import matplotlib.pyplot as plt
import os

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
    #print("batch_predictions: ", batch_predictions.shape)
    #print("classification_prediction: ", classification_prediction.shape)
    # batch_predictions = output (batch, output_dim, range_bin, angle_bin) 
    # [:, 0, :, :]: classification, [:, 1, :, :]: range,  [:, 2, :, :]: angle
    classification_label = batch_labels[:, 0,:, :].contiguous().flatten()
    #print("batch_labels: ", batch_labels.shape)
    #print("classification_label: ", classification_label.shape)

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
    #print("batch_labels[0,2:]: ", batch_labels[0,2:])
    P = batch_predictions[:,1:] # only take range and angle, exclude classificstion = batch_predictions[:, 0]
    #print("batch_predictions[0,2:]: ", batch_predictions[0,2:])
    M = batch_labels[:,0].unsqueeze(1)
    #print("batch_labels: ", batch_labels.shape, "batch_predictions: ", batch_predictions.shape)
    #print("T: ", T.shape, "P: ", P.shape, "M: ", M.shape)

    if(param['regression']=='SmoothL1Loss'):
        reg_loss_fct = nn.SmoothL1Loss(reduction='sum')
    else:
        reg_loss_fct = nn.L1Loss(reduction='sum')
    
    regression_loss = reg_loss_fct(P*M,T)
    NbPts = M.sum()
    if(NbPts>0):
        regression_loss/=NbPts

    return classification_loss,regression_loss


def MVloss(batch_predictions):
    plt.switch_backend('Agg')
    batch_predictions_RA = batch_predictions['RA']
    batch_predictions_RD = batch_predictions['RD']

    rd_range_probs = batch_predictions_RD[:, 0, :, :]
    ra_range_probs = batch_predictions_RA[:, 0, :, :]

    rd_range_prob = torch.max(rd_range_probs, dim=2, keepdim=True)[0]

    ra_range_prob = torch.max(ra_range_probs, dim=2, keepdim=True)[0]
    MVLoss = nn.SmoothL1Loss(reduction='mean')
    loss = MVLoss(rd_range_prob, ra_range_prob)
    

    # # Assuming rd_range_probs is a 2D tensor (range, Doppler/angle) after max
    # plt.imshow(rd_range_probs[0, :, :].detach().cpu().numpy().copy())  # Without rotation
    # plt.title("RD Without Rotation")
    # filepath = os.path.join('/imec/other/dl4ms/chu06/RD_Without_Rotation.png')
    # plt.savefig(filepath)
    # print(f'Plot saved to {filepath}')
    # plt.close()     

    # # After rotation
    # rd_rotated = torch.rot90(rd_range_probs, 2, [1, 2])
    # plt.imshow(rd_rotated[0, :, :].detach().cpu().numpy().copy())  # After rotation
    # plt.title("RD After Rotation")
    # filepath = os.path.join('/imec/other/dl4ms/chu06/RD_After_Rotation.png')
    # plt.savefig(filepath)
    # print(f'Plot saved to {filepath}')
    # plt.close()    

    # # Assuming rd_range_probs is a 2D tensor (range, Doppler/angle) after max
    # plt.imshow(ra_range_probs[0, :, :].detach().cpu().numpy().copy())  # Without rotation
    # plt.title("RA Without Rotation")
    # filepath = os.path.join('/imec/other/dl4ms/chu06/RA_Without_Rotation.png')
    # plt.savefig(filepath)
    # print(f'Plot saved to {filepath}')
    # plt.close()   

    # # After rotation
    # rd_rotated = torch.rot90(rd_range_probs, 2, [1, 2])
    # plt.imshow(ra_range_probs[0, :, :].detach().cpu().numpy().copy())  # After rotation
    # plt.title("RA After Rotation")
    # filepath = os.path.join('/imec/other/dl4ms/chu06/RA_After_Rotation.png')
    # plt.savefig(filepath)
    # print(f'Plot saved to {filepath}')
    # plt.close()   

    #raise Exception("finishing checking RA RD output")
    return loss
# class MVLoss(nn.Module):
#     """
#     Compute the Unsupervised Coherence Loss

#     PARAMETERS
#     ----------
#     global_weight: float
#         Global weight to apply on the entire loss
#     """

#     def __init__(self, global_weight: float = 1.) -> None:
#         super(MVLoss, self).__init__()
#         self.global_weight = global_weight
#         self.MVLoss = torch.nn.HuberLoss(reduction ='mean')
#         #self.MVLoss = nn.MSELoss() #instead?
#     def forward(self, rd_input: torch.Tensor,
#                 ra_input: torch.Tensor) -> torch.Tensor:
#         """Forward pass to compute the loss between the two predicted view masks"""
#         rd_softmax = nn.Softmax(dim=1)(rd_input)
#         ra_softmax = nn.Softmax(dim=1)(ra_input)
#         
#         # This extracts the maximum probabilities along the Doppler (or angle) dimension.
#         rd_range_probs = torch.max(rd_softmax, dim=3, keepdim=True)[0] # dim=3: This specifies that you want to find the maximum values across the 4th dimension 
                                                                         # (which could represent Doppler or angle bins, depending on the data structure).
                                                                         # keepdim=True: This ensures that the output tensor keeps the same number of dimensions as the input tensor.

#         # Rotate RD Range vect to match zero range
#         rd_range_probs = torch.rot90(rd_range_probs, 2, [2, 3])
#         ra_range_probs = torch.max(ra_softmax, dim=3, keepdim=True)[0]
#         loss = self.MVLoss(rd_range_probs, ra_range_probs)
#         loss = self.global_weight*loss
#         return loss