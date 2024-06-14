import os
import json
import argparse
import torch
import numpy as np
from model.FFTRadNet_ViT import FFTRadNet_ViT
from model.FFTRadNet_ViT import FFTRadNet_ViT_ADC
from dataset.dataset import RADIal
#from dataset.encoder import ra_encoder
import cv2
from utils.util import DisplayHMI

from dataset.matlab_dataset import MATLAB
from dataset.encoder_modi import ra_encoder
import matplotlib.pyplot as plt
from loss import pixor_loss
import itertools # for resume the iteration from a specific point

def main(config, checkpoint_filename,difficult):

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

        dataset = RADIal(root_dir = config['dataset']['root_dir'],
                            statistics= config['dataset']['statistics'],
                            encoder=enc.encode,
                            difficult=True,perform_FFT=config['data_mode'])
        
        # dataset = MATLAB(root_dir = '/imec/other/ruoyumsc/users/chu/matlab-radar-automotive/simulation_data_16rx/', #  simulation_data_DDA
        #                  statistics= config['dataset']['statistics'], 
        #                  encoder=enc.encode)

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

        dataset = RADIal(root_dir = config['dataset']['root_dir'],
                            statistics= config['dataset']['statistics'],
                            encoder=enc.encode,
                            difficult=True,perform_FFT='ADC')


    net.to('cuda')

    # Load the model
    dict = torch.load(checkpoint_filename)
    net.load_state_dict(dict['net_state_dict'])
    net.eval()


    # Define the directory where you want to save the plots
    save_dir = "./plot/"
    os.makedirs(save_dir, exist_ok=True)
    lowest_loss = float('inf') # 13.1834 # index 4343 
    lowest_loss_index = -1
    lowest_loss_data = None
    intermediate = None

    start_index = 0
    for idx in range(start_index, len(dataset)):
        data = dataset[idx]
    #for idx, data in enumerate(dataset): 
        #print(idx,len(dataset))
        # data is composed of [radar_FFT, segmap,out_label,box_labels,image]
        inputs = torch.tensor(data[0]).permute(2,0,1).to('cuda').unsqueeze(0)
        with torch.set_grad_enabled(False):
            inputs = inputs.float()
            outputs = net(inputs)

            if config['data_mode'] == 'ADC':
                intermediate = net.DFT(inputs).detach().cpu().numpy()[0]

            else:
                intermediate = None

        
        #print('outputs[Detection]', outputs['Detection'].shape)
        label_map = torch.tensor(data[2]).to('cuda').float().unsqueeze(0)
        #print('label_map shape: ', label_map.shape)
        classif_loss,reg_loss = pixor_loss(outputs['Detection'], label_map,config['losses'])
        loss = classif_loss + reg_loss

        if classif_loss < lowest_loss:
            lowest_loss = classif_loss
            lowest_loss_index = idx
            lowest_loss_data = data
            print('classif_loss: ', classif_loss)
            print('reg_loss: ', reg_loss)

            # Save the lowest_loss_data to a .npy file
            print('lowest_loss_index: ', lowest_loss_index)
            print("lowest_loss: ", lowest_loss)
            np.savez('/imec/other/ruoyumsc/users/chu/data/lowest_clssifloss_data.npy', *lowest_loss_data)

            hmi = DisplayHMI(data[4], data[0],outputs,enc,config,intermediate)

            # cv2.imshow('FFTRadNet',hmi)
            save_path = os.path.join(save_dir, f'FFTRadNet_image_{idx}.jpg')
            cv2.imwrite(save_path, hmi)
            print("plot save to ", save_path)

        # break


        #rd_plot(inputs[0, 0, :, :]+inputs[0, 16, :, :]*1j, idx)

        # # Press Q on keyboard to  exit
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break

        # # Plotting using matplotlib
        # plt.figure(figsize=(10, 6))
        # plt.imshow(hmi, aspect='auto', cmap='viridis')  # Assuming `hmi` is an image or 2D array you want to plot
        # plt.title("FFTRadNet Output")
        # plt.axis('off')

        # # Save the plot
        # plot_filename = f"plot_{idx}.png"
        # plot_filepath = os.path.join(save_dir, plot_filename)
        # plt.savefig(plot_filepath, bbox_inches='tight', pad_inches=0)
        # plt.close()
        # print(f"Plot saved to {save_path}")

        
        #print('output size: ', outputs['Detection'].shape) 1, 3, 128, 224
        #print('label map size: ',  torch.tensor(data[2]).to('cuda').float().shape) 3 ,128, 224
        # plot output
        # if (idx >= 50):
        #     # plot the prediction and ground truth, pixel occupied with vehicle (RA coordinate) 
        #     detection_plot(outputs['Detection'], label_map, idx)
        #     matrix_plot(outputs['Detection'], label_map, idx)

    # if lowest_loss_data is not None:
    #     print(f'The file with the lowest loss is at index {lowest_loss_index} with a loss of {lowest_loss}')

    #     data = dataset[4343]

    #     # Plot the data with the lowest loss
    #     hmi = DisplayHMI(data[4], data[0],outputs,enc,config,intermediate)

    #     # cv2.imshow('FFTRadNet',hmi)
    #     save_path = os.path.join(save_dir, f'FFTRadNet_image_{idx}.jpg')
    #     cv2.imwrite(save_path, hmi)
    # else:
    #     print('No valid data was found to plot.')    

    #cv2.destroyAllWindows()

### plot detection (classification)
def detection_plot(predictions, labels, idx):
    prediction = predictions[0, 0, :, :]
    #print(prediction[0, 0:10, 0])
    #target_prediction = (prediction > 0.5).float()
    label = labels[0, :, :]
    # Specify the directory to save the plot
    directory = './plot/'

    target_num = 0
    


    # Create a figure
    plt.figure(figsize=(6, 6))

    # Plot pre: Red points
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            #if pre[i, j] == 1:
            if prediction[i, j] > 0.5:
                target_num += 1
                #print("predict target!!!")
                plt.scatter(j, i, color='red', s=1, label='prediction' if i == 0 and j == 0 else "")
    print('number of targets in the prediction', target_num)
    # Plot lab: Blue points
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i, j] == 1:
                plt.scatter(j, i, color='blue', s=1, label='ground truth' if i == 0 and j == 0 else "")

    # Set plot limits and labels
    plt.xlim(0, prediction.shape[1])
    plt.ylim(0, prediction.shape[0])
    #plt.gca().invert_yaxis()  # To match matrix indexing
    plt.xlabel('angle Index')
    plt.ylabel('range Index')
    plt.title('Comparison of prediction and labels')
        
    # Save the plot with an incrementally named file
    filepath = os.path.join(directory, f'plot_{idx}.png')
    plt.savefig(filepath)
    print(f'Plot saved to {filepath}')

    # Close the plot to free up memory
    plt.close()        


def matrix_plot(predictions, labels, idx):
    directory = './plot/'
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    prediction = predictions[0, 0, :, :].detach().cpu().numpy().copy()
    #target_prediction = (prediction > 0.5).float()
    label = labels[0, :, :].detach().cpu().numpy().copy()
    #print('label shape', label.shape)

    m1 = axs[0].imshow(prediction, cmap='magma', interpolation='none')
    axs[0].set_title('prediction')

    fig.colorbar(m1, ax=axs[0])

    # Plot the second matrix
    m2 = axs[1].imshow(label, cmap='magma', interpolation='none', vmin=0.0, vmax=1.0)
    axs[1].set_title('label')

    fig.colorbar(m2, ax=axs[1])

    # Save the plot with an incrementally named file
    filepath = os.path.join(directory, f'matrix_plot_{idx}.png')
    plt.savefig(filepath)
    print(f'Plot saved to {filepath}')

    # Close the plot to free up memory
    plt.close()  
# check input        
def rd_plot(data, idx): 
    directory = './plot/'
    data_ = 20* np.log10(np.abs(data.detach().cpu().numpy().copy()))
    

    plt.figure(figsize=(6, 6))
    plt.imshow(data_)
    # Save the plot with an incrementally named file
    filepath = os.path.join(directory, f'RDplot_{idx}.png')
    plt.savefig(filepath)
    print(f'Plot saved to {filepath}')

    # Close the plot to free up memory
    plt.close() 




if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet test')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config, args.checkpoint,args.difficult)
