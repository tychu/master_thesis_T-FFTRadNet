import numpy as np
import sys
        
class ra_encoder():
    def __init__(self, geometry, statistics,regression_layer = 2):
        
        self.geometry = geometry
        self.statistics = statistics
        self.regression_layer = regression_layer

        self.INPUT_DIM = (geometry['ranges'][0],geometry['ranges'][1],geometry['ranges'][2])
        self.OUTPUT_DIM = (regression_layer + 1,self.INPUT_DIM[0] // 4 , self.INPUT_DIM[1] // 4 , self.INPUT_DIM[2])

      

    def encode(self,labels):
        map = np.zeros(self.OUTPUT_DIM )
        #print("encoder map shape: ", map.shape)

        for lab in labels:
            # [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix]

            if(lab[0]==-1):
                continue

            #range_bin = int(np.clip(lab[0]/self.geometry['resolution'][0]/4,0,self.OUTPUT_DIM[1]))
            range_bin = int(np.clip(lab[0]/self.geometry['resolution'][0]/4,0,self.OUTPUT_DIM[1]-1))
            range_mod = lab[0] - range_bin*self.geometry['resolution'][0]*4

            # ANgle and deg
            #angle_bin = int(np.clip(np.floor(lab[1]/self.geometry['resolution'][1]/4 + self.OUTPUT_DIM[2]/2),0,self.OUTPUT_DIM[2]))
            angle_bin = int(np.clip(np.floor(lab[1]/self.geometry['resolution'][1]/4 + self.OUTPUT_DIM[2]/2),0,self.OUTPUT_DIM[2]-1))
            angle_mod = lab[1] - (angle_bin- self.OUTPUT_DIM[2]/2)*self.geometry['resolution'][1]*4 

            # velocirty
            vel_bin = int(np.clip(np.floor(lab[2]/self.geometry['resolution'][2] + self.OUTPUT_DIM[3]/2),0,self.OUTPUT_DIM[3]-1))
            vel_mod = lab[2] - (vel_bin- self.OUTPUT_DIM[3]/2)*self.geometry['resolution'][2]

            #print("velocity: ", lab[2])
            #print("vel_bin: ", vel_bin)
            
            if(self.geometry['size']==1):
                
                map[0,range_bin,angle_bin,vel_bin] = 1
                map[1,range_bin,angle_bin,vel_bin] = (range_mod - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                map[2,range_bin,angle_bin,vel_bin] = (angle_mod - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
                map[3,range_bin,angle_bin,vel_bin] = (vel_mod - self.statistics['reg_mean'][2]) / self.statistics['reg_std'][2]
            else:
                #print("in self.geometry['size'] != 1")

                s = int((self.geometry['size']-1)/2) #size=3 (3-1)/2=1
                #print("s: ", s)
                r_lin = np.linspace(self.geometry['resolution'][0]*s, -self.geometry['resolution'][0]*s,
                                    self.geometry['size'])*4
                a_lin = np.linspace(self.geometry['resolution'][1]*s, -self.geometry['resolution'][1]*s,
                                    self.geometry['size'])*4
                v_lin = np.linspace(self.geometry['resolution'][2]*s, -self.geometry['resolution'][2]*s,
                                    self.geometry['size'])
                #print("r_lin: ", r_lin, "a_lin: ", a_lin)
                # np.linspace(): generate array evenly spaced number over a specified interval
                px_a, px_r, px_v = np.meshgrid(a_lin, r_lin, v_lin)
                #print("px_a: ", px_a, "px_r: ", px_r)

                # # Define the range and angle bin limits
                # range_start = range_bin - s
                # range_end = range_bin + s + 1
                # angle_start = angle_bin - s
                # angle_end = angle_bin + s + 1

                # if range_start < 0 or range_end > self.OUTPUT_DIM[1] or angle_start < 0 or angle_end > self.OUTPUT_DIM[2]:
                #     raise ValueError("Range or angle bins are out of bounds")
                
                # Define the range and angle bin limits
                # range_start = max(range_bin - s, 0)
                # range_end = min(range_bin + s + 1, self.OUTPUT_DIM[1])
                # angle_start = max(angle_bin - s, 0)
                # angle_end = min(angle_bin + s + 1, self.OUTPUT_DIM[2])
                # vel_start = max(vel_bin - s, 0)
                # vel_end = min(vel_bin + s + 1, self.OUTPUT_DIM[3])

                map = update_map(map, range_bin, angle_bin, vel_bin, s, px_r, px_a, px_v, range_mod, angle_mod, vel_mod, self.statistics, self.OUTPUT_DIM)

                
        return map
        
    def decode(self,map,threshold):
       
        range_bins,angle_bins,vel_bins = np.where(map[0,:,:,:]>=threshold)

        coordinates = []

        for range_bin,angle_bin,vel_bin in zip(range_bins,angle_bins,vel_bins):


            R = range_bin*4*self.geometry['resolution'][0] + map[1,range_bin,angle_bin,vel_bin] * self.statistics['reg_std'][0] + self.statistics['reg_mean'][0]

            A = (angle_bin-self.OUTPUT_DIM[2]/2)*4*self.geometry['resolution'][1] + map[2,range_bin,angle_bin,vel_bin] * self.statistics['reg_std'][1] + self.statistics['reg_mean'][1]
            V = (vel_bin-self.OUTPUT_DIM[3]/2)*self.geometry['resolution'][2] + map[3,range_bin,angle_bin,vel_bin] * self.statistics['reg_std'][2] + self.statistics['reg_mean'][2]

            C = map[0,range_bin,angle_bin,vel_bin]
            #  map[1,range_bin,angle_bin]: Model prediction on range, reg head
            #  map[2,range_bin,angle_bin]: Model prediction on angle, reg head
        
            coordinates.append([R,A,V,C])
            #print("R: ", R.shape, "A: ", A.shape, "V: ", V.shape, "C: ", C.shape)
            #print(coordinates.shape)
            
        #print("coordinates:", len(coordinates))
        return coordinates

###### functions for encoder
# Function to calculate start and end indices
def get_start_end_indices(bin_idx, s, output_dim):
    start = max(bin_idx - s, 0)
    end = min(bin_idx + s + 1, output_dim)
    return start, end

# Function to get sliced px data
def get_sliced_data(bin_idx, s, px_r, px_a, px_v, output_dim, dim_type):
    if dim_type == "range":
        if bin_idx >= s and bin_idx < (output_dim - s):
            return px_r, px_a, px_v  # Central case, no slicing needed
        elif bin_idx < s:
            return px_r[s - bin_idx:, :, :], px_a[s - bin_idx:, :, :], px_v[s - bin_idx:, :, :]  # Left boundary case
        else:
            return px_r[:s + (output_dim - bin_idx), :, :], px_a[:s + (output_dim - bin_idx), :, :], px_v[:s + (output_dim - bin_idx), :, :]  # Right boundary case
    elif dim_type == "angle":
        if bin_idx >= s and bin_idx < (output_dim - s):
            return px_r, px_a, px_v  # Central case, no slicing needed
        elif bin_idx < s:
            return px_r[:, s - bin_idx:, :], px_a[:, s - bin_idx:, :], px_v[:, s - bin_idx:, :]  # Left boundary case
        else:
            return px_r[:, :s + (output_dim - bin_idx), :], px_a[:, :s + (output_dim - bin_idx), :], px_v[:, :s + (output_dim - bin_idx), :]  # Right boundary case
    elif dim_type == "velocity":
        if bin_idx >= s and bin_idx < (output_dim - s):
            return px_r, px_a, px_v  # Central case, no slicing needed
        elif bin_idx < s:
            return px_r[:, :, s - bin_idx:], px_a[:, :, s - bin_idx:], px_v[:, :, s - bin_idx:]  # Left boundary case
        else:
            return px_r[:, :, :s + (output_dim - bin_idx)], px_a[:, :, :s + (output_dim - bin_idx)], px_v[:, :, :s + (output_dim - bin_idx)]  # Right boundary case

# Function to update the map based on provided indices and slices
def update_map_slice(map, indices, px_r_slice, px_a_slice, px_v_slice, range_mod, angle_mod, vel_mod, reg_mean, reg_std):
    range_start, range_end, angle_start, angle_end, vel_start, vel_end = indices

    map[0, range_start:range_end, angle_start:angle_end, vel_start:vel_end] = 1
    map[1, range_start:range_end, angle_start:angle_end, vel_start:vel_end] = (px_r_slice + range_mod - reg_mean[0]) / reg_std[0]
    map[2, range_start:range_end, angle_start:angle_end, vel_start:vel_end] = (px_a_slice + angle_mod - reg_mean[1]) / reg_std[1]
    map[3, range_start:range_end, angle_start:angle_end, vel_start:vel_end] = (px_v_slice + vel_mod - reg_mean[2]) / reg_std[2]

    return map

# Function to handle each case for range, angle, and velocity bins
def process_map_case(map, range_bin, angle_bin, vel_bin, s, px_r, px_a, px_v, range_mod, angle_mod, vel_mod, reg_mean, reg_std, output_dim):
    # Get start and end indices for range, angle, and velocity
    range_start, range_end = get_start_end_indices(range_bin, s, output_dim[1])
    angle_start, angle_end = get_start_end_indices(angle_bin, s, output_dim[2])
    vel_start, vel_end = get_start_end_indices(vel_bin, s, output_dim[3])

    # Slice the px_r, px_a, and px_v data based on range, angle, and velocity bin cases
    px_rr, px_ar, px_vr = get_sliced_data(range_bin, s, px_r, px_a, px_v, output_dim[1], 'range')
    px_ra, px_aa, px_va = get_sliced_data(angle_bin, s, px_rr, px_ar, px_vr, output_dim[2], 'angle')
    px_r_slice, px_a_slice, px_v_slice = get_sliced_data(vel_bin, s, px_ra, px_aa, px_va, output_dim[3], 'velocity')

    # Update the map for the current slice
    map = update_map_slice(map, (range_start, range_end, angle_start, angle_end, vel_start, vel_end),
                     px_r_slice, px_a_slice, px_v_slice, range_mod, angle_mod, vel_mod, reg_mean, reg_std)
    
    return map

# Main function that handles map updates
def update_map(map, range_bin, angle_bin, vel_bin, s, px_r, px_a, px_v, range_mod, angle_mod, vel_mod, statistics, output_dim):
    reg_mean = statistics['reg_mean']
    reg_std = statistics['reg_std']

    # # Get case indicators for range, angle, and velocity bins
    # range_case = 0 if s <= range_bin < (output_dim[1] - s) else (-1 if range_bin < s else 1)
    # angle_case = 0 if s <= angle_bin < (output_dim[2] - s) else (-1 if angle_bin < s else 1)
    # vel_case = 0 if s <= vel_bin < (output_dim[3] - s) else (-1 if vel_bin < s else 1)

    # Process map for all cases
    map = process_map_case(map, range_bin, angle_bin, vel_bin, s, px_r, px_a, px_v,
                     range_mod, angle_mod, vel_mod, reg_mean, reg_std, output_dim)
    
    return map