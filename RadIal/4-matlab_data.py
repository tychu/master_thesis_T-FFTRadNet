import scipy.io as sio
import numpy as np
import os
import pandas as pd 

####
# loading matlab simulation data
#####

variable_name = 'radar_data_cube_4d'
root_dir = '/imec/other/ruoyumsc/users/chu/matlab-radar-automotive/simulation_data_16rx/'

# path to directory containing MATLAB files
#mat_dir = '/imec/other/ruoyumsc/users/chu/matlab-radar-automotive/simulation_data_DDA/RD_cube/'
mat_dir = os.path.join(root_dir, 'RD_cube/')





filename = os.path.join(mat_dir, "RD_cube_1.mat")  # Assuming the files are named data_1.npy to data_100.npy
mat_data = sio.loadmat(filename)[variable_name]
print(mat_data.shape) # (512, 256, 1, 16)




num_sample = 100
num_tx = 16
# compute the statistic
####
# Initialize an array to store the cube data
cube_data = np.zeros((num_sample, 512, 256, 2*num_tx)) # stack real and imaginary part 16*2=32

# Read the cube data from files
for i in range(num_sample):
    filename = os.path.join(mat_dir, f"RD_cube_{i+1}.mat")  # Assuming the files are named data_1.npy to data_100.npy
    mat_data = sio.loadmat(filename)[variable_name]
    combine_data_list = []

    for cube_idx in range(mat_data.shape[-1]):
        cube = mat_data[:, :, :, cube_idx]
        # combined the column of each matrix
        combined_data_cube = np.concatenate([cube[:, i, :] for i in range(cube.shape[1])], axis=1)
    
        # append the combined data to the list
        combine_data_list.append(combined_data_cube)

    final_data_3d = np.stack(combine_data_list, axis=-1)


    real_part = np.real(final_data_3d)
    imag_part = np.imag(final_data_3d)
    cube_data[i] = np.concatenate((real_part, imag_part), axis=-1)


# Calculate the mean for each cube along the last axis (axis=3)
cube_means = np.mean(cube_data, axis=(1, 2))
print('the mean for each cube along the last axis')
print(cube_means.shape)

# Calculate the mean value array across all cubes
overall_mean = np.mean(cube_means, axis=0)
overall_std = np.std(cube_means, axis=0)

# Print the mean value array
print("Overall mean value array:")
print(overall_mean)
print("Overall std value array:")
print(overall_std)  

# Overall mean value array:
# [-8.75710095e-18  1.96067121e-15 -3.38213615e-16  1.40254561e-16
#  -1.82988282e-15 -4.35507750e-17  4.85082894e-17  2.72855740e-16
#   1.05539600e-15 -4.41893158e-16  5.64096838e-17 -8.69088737e-16
#   7.02523434e-16  2.47149957e-15  5.17929341e-16 -8.64924452e-16
#  -1.84642883e-16 -4.28427096e-16  2.41588027e-15  8.48110102e-16
#  -1.03205801e-15 -3.85859964e-16 -1.71797255e-16  6.57956220e-17
#   3.71012493e-16 -1.72279915e-15 -4.34181635e-16 -7.84614344e-16
#   6.44403151e-16 -1.41970515e-15  2.55493326e-16  4.89999751e-16]
# Overall std value array:
# [3.01317133e-16 1.96301148e-14 3.95032439e-15 1.13825836e-15
#  1.80725822e-14 5.47412066e-16 6.79764969e-16 2.48464066e-15
#  1.15818531e-14 3.87210843e-15 3.73108950e-16 9.28098168e-15
#  6.39689733e-15 2.48599808e-14 4.87278791e-15 8.18394829e-15
#  1.56793279e-15 4.17721280e-15 2.36336924e-14 8.32084867e-15
#  1.02511997e-14 3.45741830e-15 1.56371053e-15 7.48021873e-16
#  3.56001920e-15 1.68589488e-14 4.89698692e-15 7.17424173e-15
#  6.03824994e-15 1.48463729e-14 2.32290987e-15 4.22575217e-15]

## Ntx = 2
# Overall mean value array:
# [ 1.29290567e-17 -7.21034560e-17 -5.68744813e-17 -1.22157059e-18]
# Overall std value array:
# [7.88506969e-16 1.31454054e-15 5.46832158e-16 6.73096496e-16]

 ######################
 # convert the labels #
 ######################

# path to directory containing CVS files
#csv_dir = '/imec/other/ruoyumsc/users/chu/matlab-radar-automotive/simulation_data_DDA/ground_truth/'
csv_dir = os.path.join(root_dir, 'ground_truth/')
# List MATLAB files in directory
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
print(len(csv_files))
# load and combine arrarys from MATLAB files
combined_label_df = pd.concat([pd.read_csv(os.path.join(csv_dir, f)) for f in csv_files], ignore_index=True)
# calculate mean and std
#columns_means = combined_label_df.mean()
#columns_stds = combined_label_df.std()



resolution= [0.201171875,0.2]
ranges = [512,896,1]
regression_layer = 2
INPUT_DIM = (ranges[0], ranges[1], ranges[2])
OUTPUT_DIM = (regression_layer + 1,INPUT_DIM[0] // 4 , INPUT_DIM[1] // 4 )


# range
#range_bin = int(np.clip(combined_label_df['range'].to_numpy()/resolution[0]/4,0, OUTPUT_DIM[1]-1))
#range_mod = combined_label_df['range'].to_numpy() - range_bin*resolution[0]*4
#combined_label_df['range_processed'] = range_mod

# Compute the 'range_bin' for each element  512//4=128=OUTPUT_DIM[1]=INPUT_DIM[0] // 4
combined_label_df['range_bin'] = (combined_label_df['range'] / resolution[0] / 4).clip(0, OUTPUT_DIM[1] - 1).astype(int)
combined_label_df['range_inter_compute'] = combined_label_df['range_bin'] * resolution[0] * 4
# Compute 'range_mod' for each element
combined_label_df['range_mod'] = combined_label_df['range'] - combined_label_df['range_bin'] * resolution[0] * 4


# ANgle and deg
# Compute the 'angle_bin' for each element 896//4=224=OUTPUT_DIM[2]=INPUT_DIM[1] // 4
#combined_label_df['angle_bin'] = (np.floor(combined_label_df['azimuth'].to_numpy() / resolution[1] / 4 + OUTPUT_DIM[2] / 2)).clip(0, OUTPUT_DIM[2] - 1).astype(int)
combined_label_df['angle_bin'] = (np.floor(combined_label_df['azimuth'].to_numpy() / resolution[1] / 4 + OUTPUT_DIM[2]/2)).clip(0, OUTPUT_DIM[2] - 1).astype(int)
combined_label_df['angle_inter_compute'] = (combined_label_df['angle_bin'] - OUTPUT_DIM[2]/2) * resolution[1] * 4

# Compute 'angle_mod' for each element
#combined_label_df['angle_mod'] = combined_label_df['azimuth'].to_numpy() - (combined_label_df['angle_bin'] - OUTPUT_DIM[2] / 2) * resolution[1] * 4
combined_label_df['angle_mod'] = combined_label_df['azimuth'].to_numpy() - (combined_label_df['angle_bin'] - OUTPUT_DIM[2] / 2) * resolution[1] * 4


columns_means = combined_label_df.mean()
columns_stds = combined_label_df.std()

print("labels overall mean value array:")
print(columns_means)
print("labels overall std value array:")
print(columns_stds) 

print(combined_label_df.iloc[:10, :])

# labels overall mean value array:
# range                   53.487021
# azimuth                 -3.249200
# doppler                128.477000
# range_bin               65.983000
# range_inter_compute     53.095695
# range_mod                0.391326
# angle_bin              107.560000
# angle_inter_compute     -3.552000
# angle_mod                0.302800
# dtype: float64
# labels overall std value array:
# range                  20.404556
# azimuth                49.535506
# doppler                 0.499721
# range_bin              25.375591
# range_inter_compute    20.419421
# range_mod               0.229499
# angle_bin              61.902523
# angle_inter_compute    49.522018
# angle_mod               0.225661
# dtype: float64
