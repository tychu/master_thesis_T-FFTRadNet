from torch.utils.data import Dataset
import numpy as np
import os
import torch
from torchvision.transforms import Resize,CenterCrop
import torchvision.transforms as transform
import pandas as pd
from PIL import Image
import scipy.io as sio

class MATLAB(Dataset):

    def __init__(self, root_dir,folder_dir, statistics=None,encoder_ra=None,encoder_rd=None,difficult=False,perform_FFT='Default'):

        self.root_dir = root_dir
        self.folder_dir = folder_dir
        self.statistics = statistics
        self.ra_encoder = encoder_ra
        self.rd_encoder = encoder_rd
        self.perform_FFT = perform_FFT #'Default'

        #self.labels = pd.read_csv(os.path.join(root_dir,'ground_truth/ground_truth.csv')).to_numpy()
        #csv_file = os.path.join(self.root_dir, 'ground_truth/ground_truth.csv')
        #self.labels = pd.read_csv(csv_file).to_numpy()
        #print('check labels length')
        #print(len(self.labels))

        #self.mat_files = [f for f in os.listdir(os.path.join(self.root_dir, self.folder_dir, 'RD_cube/')) if f.endswith('.mat')]
        #print("dataset len mat file: ", len(self.mat_files))
        self.csv_files = [f for f in os.listdir(os.path.join(self.root_dir, self.folder_dir, 'ground_truth/')) if f.endswith('.csv')]
        #print("dataset len csv: ", len(self.csv_files))

        #self.image_files = [f for f in os.listdir(os.path.join(self.root_dir, 'camera/')) if f.endswith('.jpg')]
        
       
        # Keeps only easy samples
        #if(difficult==False):
        #    ids_filters=[]
        #    ids = np.where( self.labels[:, -1] == 0)[0]
        #    ids_filters.append(ids)
        #    ids_filters = np.unique(np.concatenate(ids_filters))
        #    self.labels = self.labels[ids_filters]


        # Gather each input entries by their sample id
        #self.unique_ids = np.unique(self.labels[:,0])
        #self.label_dict = {}
        #for i,ids in enumerate(self.unique_ids):
        #    sample_ids = np.where(self.labels[:,0]==ids)[0]
        #    self.label_dict[ids]=sample_ids
        #self.sample_keys = list(self.label_dict.keys())
        
        # for segmap
        #self.resize = Resize((256,224), interpolation=transform.InterpolationMode.NEAREST)
        #self.crop = CenterCrop((512,448))


    def __len__(self):
        #return len(self.label_dict)
        print("dataset len: ", len(self.csv_files))
        return len(self.csv_files)
        #return len(self.labels)

    #def __getitem__(self, index):
    def __getitem__(self, idx):


        # print("csv file: ", len(self.csv_files))
        # print("mat files: ", len(self.mat_files))
        # print(f"Fetching index: {idx}")
        #if idx >= len(self.csv_files) or idx >= len(self.mat_files):
        #    raise IndexError(f"Index {idx} out of range for file lists. csv file: {len(self.csv_files)}, mat file: {len(self.mat_files)}")
        
        # Get the sample id
        #sample_id = self.sample_keys[index] 

        # From the sample id, retrieve all the labels ids
        #entries_indexes = self.label_dict[sample_id]

        # Get the objects labels
        #box_labels = self.labels[entries_indexes]
        
        #box_labels = self.labels

        # Labels contains following parameters:
        # x1_pix	y1_pix	x2_pix	y2_pix	laser_X_m	laser_Y_m	laser_Z_m radar_X_m	radar_Y_m	radar_R_m

        # format as following [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix]
        #box_labels = box_labels[:,[10,11,12,5,6,7,1,2,3,4]].astype(np.float32) 

        
    
        idx_adj = idx +1
        # load lables
        csv_file = os.path.join(self.root_dir, self.folder_dir, 'ground_truth/',f"ground_truth_{idx_adj}.csv")
        #print("csv file indx :", csv_file)
        #csv_file = os.path.join(self.root_dir, 'ground_truth/', self.csv_files[idx])
        box_labels = pd.read_csv(csv_file).to_numpy()
        #print("box_labels: ", box_labels.shape)


        ######################
        #  Encode the labels #
        ######################        
        #out_label=[]
        if(self.ra_encoder!=None and self.rd_encoder!=None):
            #out_label = self.encoder(box_labels).copy()
            out_label_ra = self.ra_encoder(box_labels).copy()
            out_label_rd = self.rd_encoder(box_labels).copy()
            print("out_label_rd: ", out_label_rd.shape)
            #raise Exception("finishing checking encoder")
        #out_label = box_labels      

        
        # Read the Radar data
        if self.perform_FFT == 'ADC':
            # Utilize Raw ADC
            radar_name = os.path.join(self.root_dir, self.folder_dir, 'ADC_cube/',f"ADC_cube_{idx_adj}.mat")
            mat_input_4d = sio.loadmat(radar_name)['radar_data_cube_4d']
            #print("Shape of raw ADC:", mat_input_4d.shape)
            radar_FFT = mat_input_4d[:, :, 0, :]
        else:
            # Read the Radar FFT data
            #radar_name = os.path.join(self.root_dir,'radar_FFT',"fft_{:06d}.npy".format(sample_id))
            mat_file = os.path.join(self.root_dir, self.folder_dir, 'RD_cube/',f"RD_cube_{idx_adj}.mat")
            #print("mat file indx :", mat_file)
            #mat_file = os.path.join(self.root_dir, 'RD_cube/', self.mat_files[idx])
            mat_input_4d = sio.loadmat(mat_file)['radar_data_cube_4d']
            mat_input_3d = mat_input_4d[:, :, 0, :]
        
            #print("radar FFT")
            radar_FFT = np.concatenate([mat_input_3d.real,mat_input_3d.imag],axis=2)
            #print("Shape of concatenated array:", radar_FFT.shape)

            if(self.statistics is not None):
                for i in range(len(self.statistics['input_mean'])):
                    radar_FFT[...,i] -= self.statistics['input_mean'][i]
                    radar_FFT[...,i] /= self.statistics['input_std'][i]
            #if(sta_mean is not None):
            #    for i in range(len(sta_mean)):
            #        radar_FFT[...,i] -= sta_mean[i]
            #        radar_FFT[...,i] /= sta_std[i]

        # # Read the segmentation map 
        # # just read any image to work the model first
        # ## TODO: remove image from model input
        # #print("seg map")
        # #segmap_name = os.path.join(self.root_dir,'radar_Freespace',"freespace_{:06d}.png".format(sample_id))
        # segmap_name = os.path.join(self.root_dir, 'camera/', self.image_files[idx_adj])
        # #segmap_name = os.path.join(self.root_dir,'radar_Freespace/',"freespace.png")
        # segmap = Image.open(segmap_name) # [512,900]
        # # 512 pix for the range and 900 pix for the horizontal FOV (180deg)
        # # We crop the fov to 89.6deg
        # segmap = self.crop(segmap)
        # # and we resize to half of its size
        # segmap = np.asarray(self.resize(segmap))==255
        # #print("segmap : ", len(segmap))

        # # Read the camera image
        # img_name = os.path.join(self.root_dir, 'camera/', self.image_files[idx_adj])
        # #img_name = os.path.join(self.root_dir,'camera/',"image_001010.jpg")
        # image = np.asarray(Image.open(img_name))
        # #print("image : ", len(image))

        #return radar_FFT, segmap,out_label,box_labels,image
        return radar_FFT,out_label_ra, out_label_rd, box_labels