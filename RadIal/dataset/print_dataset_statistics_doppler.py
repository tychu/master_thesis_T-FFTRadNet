import sys
sys.path.insert(0, '../')

#from dataset.dataset import RADIal
#from dataset.encoder import ra_encoder
#from dataset.encoder_modi import ra_encoder
from dataset.encoder_doppler import ra_encoder
#from dataset.matlab_dataset_ddp import MATLAB
from dataset.matlab_dataset_doppler import MATLAB
import numpy as np


geometry = {    "ranges": [512,896,72],
                "resolution": [0.17,0.2, 0.5], # meter
                "size": 3}

statistics = {  "input_mean":np.zeros(32),
                "input_std":np.ones(32),
                "reg_mean":np.zeros(3),
                "reg_std":np.ones(3)}

enc = ra_encoder(geometry = geometry,
                    statistics = statistics,
                    regression_layer = 3)

# dataset = RADIal(root_dir = r"C:\Users\James-PC\James\data\RADIal",
#                        statistics=None,
#                        encoder=enc.encode,
#                        difficult=True,perform_FFT='Custom_RD')
dataset = MATLAB(root_dir = '/imec/other/dl4ms/chu06/public/data/', #replace with the correct directory
                 folder_dir = "simulation_sequential_rw_data_DDA_rawADC/",
                 statistics= None, 
                 encoder=enc.encode,perform_FFT='ADC')

reg = []
m=0
s=0
for i in range(len(dataset)):
    print(i,len(dataset))
    #radar_FFT, segmap,out_label,box_labels= dataset.__getitem__(i)
    radar_FFT,out_label,box_labels= dataset.__getitem__(i)
    print("radar_FFT: ", radar_FFT.shape)

    #data = np.reshape(radar_FFT,(512*256,32))
    data = np.reshape(radar_FFT,(512*256,4)) # rx: 4, ADC don't separate img and real

    m += data.mean(axis=0)
    s += data.std(axis=0)
    print("out_label[0]: ", out_label[0].shape)
    idy,idx,idz = np.where(out_label[0]>0) # out_label: map[0-3, r_idx, a_idx, v_idx]

    reg.append(out_label[1:,idy,idx,idz])
    print("out_label[1:,idy,idx,idz]: ", out_label[1:,idy,idx,idz])


reg = np.concatenate(reg,axis=1)
print("reg shape: ", reg.shape)

print('===  INPUT  ====')
print('mean',m/len(dataset))
print('std',s/len(dataset))
# Print mean and std separated by commas
print('mean:', ', '.join(map(str, m/len(dataset))))
print('std:', ', '.join(map(str, s/len(dataset))))

print('===  Regression  ====')
print('mean',np.mean(reg,axis=1))
print('std',np.std(reg,axis=1))
# Print mean and std separated by commas
print('mean:', ', '.join(map(str, np.mean(reg,axis=1))))
print('std:', ', '.join(map(str, np.std(reg,axis=1))))
# mean: 0.3676954382230018, 0.29876685086155963
# std: 0.6753994169817015, 0.6905939539763425

# doppler
# mean: 0.36771633332934167, 0.2988326401176297, 5.330470187592107
# std: 0.6757357195066501, 0.6906258582365226, 2.8231795887605458