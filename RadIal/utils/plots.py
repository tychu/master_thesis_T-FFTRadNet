
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import cv2

matplotlib.use('Agg')
## plots for visualization model outputs

def matrix_plot(base_dir, predictions, labels, output_map, datamode, num):
    if output_map == "RA":
        x_label = "azimuth"
    elif output_map == "RD":
        x_label = "doppler"
     
    print("in matrix plot")
    directory = os.path.join(base_dir, 'plot/')
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    prediction = predictions[0, 0, :, :]
    print("prediction: ", prediction.shape)
    #target_prediction = (prediction > 0.5).float()
    label = labels[0, 0, :, :]

    m1 = axs[0].imshow(prediction, cmap='magma', interpolation='none')
    axs[0].set_title('prediction')
    axs[0].set_ylim(0, prediction.shape[0])
    axs[0].set_xlim(0, prediction.shape[1])
    axs[0].set_xlabel(x_label)
    axs[0].set_ylabel('range')

    fig.colorbar(m1, ax=axs[0])

    # Plot the second matrix
    m2 = axs[1].imshow(label, cmap='magma', interpolation='none', vmin=0.0, vmax=1.0)
    axs[1].set_title('label')
    axs[1].set_ylim(0, label.shape[0])
    axs[1].set_xlim(0, label.shape[1])
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel('range')

    fig.colorbar(m2, ax=axs[1])

    # Save the plot with an incrementally named file
    # Check if the directory exists, if not, create it
    if not os.path.exists( directory):
        os.makedirs(directory)

    filepath = os.path.join(directory,f'matrix_plot_{datamode}_{num}.png')
    plt.savefig(filepath)
    print(f'Plot saved to {filepath}')

    # Close the plot to free up memory
    plt.close()     

def plot_prediction_with_histogram(prediction):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plotting the prediction using imshow
    m1 = axs[0].imshow(prediction, cmap='magma', interpolation='none')
    axs[0].set_title('Prediction')
    axs[0].set_ylim(0, prediction.shape[0])
    axs[0].set_xlim(0, prediction.shape[1])
    axs[0].set_xlabel('Azimuth')
    axs[0].set_ylabel('Range')
    fig.colorbar(m1, ax=axs[0])

    # Plotting the histogram of the prediction's values
    axs[1].hist(prediction.ravel(), bins=50, color='purple')
    axs[1].set_title('Histogram of Prediction Values')
    axs[1].set_xlabel('Prediction Value')
    axs[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    
### plot detection (classification)
def detection_plot(base_dir, predictions, labels, output_map, datamode, num):
    if output_map == "RA":
        x_label = "angle Index"
    elif output_map == "RD":
        x_label = "doppler Index"
    prediction = predictions[0, 0, :, :]
    
    #target_prediction = (prediction > 0.5).float()
    label = labels[0, 0, :, :]
    # Specify the directory to save the plot
    directory = os.path.join(base_dir, 'plot') #'./plot_0706_16rx_seqdata/'
    # Iterate through each matrix

    target_num = 0


    # Create a figure
    plt.figure(figsize=(6, 6))

    # Plot pre: Red points
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            #if pre[i, j] == 1:
            if prediction[i, j] >= 0.2:
                target_num += 1
                #print("predict target!!! probability: ", prediction[i, j] )
                plt.scatter(j, i, color='red', s=1, label='prediction' if i == 0 and j == 0 else "")
    #print('number of targets in the prediction', target_num)
    # Plot lab: Blue points
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i, j] == 1:
                plt.scatter(j, i, color='blue', s=1, label='ground truth' if i == 0 and j == 0 else "")

    # Set plot limits and labels
    plt.xlim(0, prediction.shape[1])
    plt.ylim(0, prediction.shape[0])
    #plt.gca().invert_yaxis()  # To match matrix indexing
    plt.xlabel(x_label)
    plt.ylabel('range Index')
    plt.title('Comparison of prediction and labels')
    
    # Save the plot with an incrementally named file
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, f'plot_{datamode}_{num}.png')
    plt.savefig(filepath)
    print(f'Plot saved to {filepath}')

    # Close the plot to free up memory
    plt.close()        



## plot histograms and also create video
def plot_histograms(predictions, epoch, histogram, batch, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    prediction = predictions[0, 0, :, :]
    plt.figure(figsize=(8, 6))
    plt.hist(prediction.ravel(), bins=50, color='purple')
    plt.title(f'Histogram of Prediction Values - Epoch {epoch}')
    plt.xlabel('Prediction Value')
    plt.ylabel('Frequency')
    plt.ylim(0, 1000)
    filepath = os.path.join(output_dir, f'histogram_{histogram}_epoch{epoch}_batch{batch}.png')
    plt.savefig(filepath)
    print(f'Plot saved to {filepath}')
    plt.close()

def create_video_from_images(image_dir, output_video, fps=2):
    #output_video = "histograms_video.mp4"
    images = sorted([img for img in os.listdir(image_dir) if img.endswith(".png")])
    frame = cv2.imread(os.path.join(image_dir, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_dir, image)))

    cv2.destroyAllWindows()
    video.release()

