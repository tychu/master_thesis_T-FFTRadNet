import numpy as np
import sys
        
class ra_encoder():
    def __init__(self, geometry, statistics,regression_layer = 2):
        
        self.geometry = geometry
        self.statistics = statistics
        self.regression_layer = regression_layer

        self.INPUT_DIM = (geometry['ranges'][0],geometry['ranges'][1],geometry['ranges'][2])
        self.OUTPUT_DIM = (regression_layer + 1,self.INPUT_DIM[0] // 4 , self.INPUT_DIM[1] // 4, self.INPUT_DIM[2])
        #print("self.INPUT_DIM: ", self.INPUT_DIM )
        #print("self.OUTPUT_DIM: ", self.OUTPUT_DIM)
        

    # def encode(self,labels):
    #     selected_dims = (self.OUTPUT_DIM[0], self.OUTPUT_DIM[1], self.OUTPUT_DIM[2])

    #     # Initialize the numpy array with the selected dimensions
    #     map = np.zeros(selected_dims)
    #     #map = np.zeros(self.OUTPUT_DIM )

    #     for lab in labels:
    #         # [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix]

    #         if(lab[0]==-1):
    #             continue

    #         #range_bin = int(np.clip(lab[0]/self.geometry['resolution'][0]/4,0,self.OUTPUT_DIM[1]))
    #         range_bin = int(np.clip(lab[0]/self.geometry['resolution'][0]/4,0,self.OUTPUT_DIM[1]-1))
    #         range_mod = lab[0] - range_bin*self.geometry['resolution'][0]*4

    #         # ANgle and deg
    #         #angle_bin = int(np.clip(np.floor(lab[1]/self.geometry['resolution'][1]/4 + self.OUTPUT_DIM[2]/2),0,self.OUTPUT_DIM[2]))
    #         angle_bin = int(np.clip(np.floor(lab[1]/self.geometry['resolution'][1]/4 + self.OUTPUT_DIM[2]/2),0,self.OUTPUT_DIM[2]-1))
    #         angle_mod = lab[1] - (angle_bin- self.OUTPUT_DIM[2]/2)*self.geometry['resolution'][1]*4 

    #         # doppler

            
            
    #         if(self.geometry['size']==1):
                
    #             map[0,range_bin,angle_bin] = 1
    #             map[1,range_bin,angle_bin] = (range_mod - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
    #             map[2,range_bin,angle_bin] = (angle_mod - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
    #         else:
    #             #print("in self.geometry['size'] != 1")

    #             s = int((self.geometry['size']-1)/2) #size=3 (3-1)/2=1
    #             #print("s: ", s)
    #             r_lin = np.linspace(self.geometry['resolution'][0]*s, -self.geometry['resolution'][0]*s,
    #                                 self.geometry['size'])*4
    #             a_lin = np.linspace(self.geometry['resolution'][1]*s, -self.geometry['resolution'][1]*s,
    #                                 self.geometry['size'])*4
    #             #print("r_lin: ", r_lin, "a_lin: ", a_lin)
    #             # np.linspace(): generate array evenly spaced number over a specified interval
    #             px_a, px_r = np.meshgrid(a_lin, r_lin)
    #             print("px_a: ", px_a.shape, "px_r: ", px_r.shape)

    #             # # Define the range and angle bin limits
    #             # range_start = range_bin - s
    #             # range_end = range_bin + s + 1
    #             # angle_start = angle_bin - s
    #             # angle_end = angle_bin + s + 1

    #             # if range_start < 0 or range_end > self.OUTPUT_DIM[1] or angle_start < 0 or angle_end > self.OUTPUT_DIM[2]:
    #             #     raise ValueError("Range or angle bins are out of bounds")
                
    #             # Define the range and angle bin limits
    #             range_start = max(range_bin - s, 0)
    #             range_end = min(range_bin + s + 1, self.OUTPUT_DIM[1])
    #             angle_start = max(angle_bin - s, 0)
    #             angle_end = min(angle_bin + s + 1, self.OUTPUT_DIM[2])
    #             #print("range_start: ", range_start, "range_end: ", range_end)
    #             #print("angle_start: ", angle_start, "angle_end: ", angle_end)

    #             # Handle all cases for angle_bin and range_bin
    #             # originally paper only cover the angle exception
    #             if angle_bin >= s and angle_bin < (self.OUTPUT_DIM[2] - s) and range_bin >= s and range_bin < (self.OUTPUT_DIM[1] - s):
    #                 print("range_start:range_end: ", range_start, range_end)
    #                 print("angle_start:angle_end: ", angle_start, angle_end)
    #                 print("r: ", ((px_r + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0])
    #                 print("a: ", ((px_a + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1])
    #                 map[0, range_start:range_end, angle_start:angle_end] = 1
    #                 map[1, range_start:range_end, angle_start:angle_end] = ((px_r + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
    #                 map[2, range_start:range_end, angle_start:angle_end] = ((px_a + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]


    #             elif angle_bin < s and range_bin >= s and range_bin < (self.OUTPUT_DIM[1] - s):
    #                 angle_slice = angle_bin + s + 1
    #                 px_r_slice = px_r[:, s - angle_bin:]
    #                 px_a_slice = px_a[:, s - angle_bin:]
    
    #                 map[0, range_start:range_end, 0:angle_slice] = 1
    #                 map[1, range_start:range_end, 0:angle_slice] = ((px_r_slice + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
    #                 map[2, range_start:range_end, 0:angle_slice] = ((px_a_slice + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
    #                 print("range_start:range_end: ", range_start, range_end)
    #                 print("angle_slice: ", angle_slice)
    #                 print("r: ", ((px_r_slice + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0])
    #                 print("a: ", ((px_a_slice + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1])

    #             elif angle_bin >= (self.OUTPUT_DIM[2] - s) and range_bin >= s and range_bin < (self.OUTPUT_DIM[1] - s):
    #                 end = s + (self.OUTPUT_DIM[2] - angle_bin)
    #                 px_r_slice = px_r[:, :end]
    #                 px_a_slice = px_a[:, :end]
    
    #                 map[0, range_start:range_end, angle_start:] = 1
    #                 map[1, range_start:range_end, angle_start:] = ((px_r_slice + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
    #                 map[2, range_start:range_end, angle_start:] = ((px_a_slice + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
    #                 print("range_start:range_end: ", range_start, range_end)
    #                 print("angle_start: ", angle_start)
    #                 print("r: ", ((px_r_slice + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0])
    #                 print("a: ", ((px_a_slice + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1])

    #             elif angle_bin >= s and angle_bin < (self.OUTPUT_DIM[2] - s) and range_bin < s:
    #                 range_slice = range_bin + s + 1
    #                 px_r_slice = px_r[s - range_bin:, :]
    #                 px_a_slice = px_a[s - range_bin:, :]
    
    #                 map[0, 0:range_slice, angle_start:angle_end] = 1
    #                 map[1, 0:range_slice, angle_start:angle_end] = ((px_r_slice + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
    #                 map[2, 0:range_slice, angle_start:angle_end] = ((px_a_slice + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
    #                 print("range_slice: ", range_slice)
    #                 print("angle_start:angle_end: ", angle_start, angle_end)
    #                 print("r: ", ((px_r_slice + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0])
    #                 print("a: ", ((px_a_slice + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1])

    #             elif angle_bin < s and range_bin < s:
    #                 angle_slice = angle_bin + s + 1
    #                 range_slice = range_bin + s + 1
    #                 px_r_slice = px_r[s - range_bin:, s - angle_bin:]
    #                 px_a_slice = px_a[s - range_bin:, s - angle_bin:]
    
    #                 map[0, 0:range_slice, 0:angle_slice] = 1
    #                 map[1, 0:range_slice, 0:angle_slice] = ((px_r_slice + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
    #                 map[2, 0:range_slice, 0:angle_slice] = ((px_a_slice + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
    #                 print("range_slice: ", range_slice)
    #                 print("angle_slice: ", angle_slice)
    #                 print("r: ", ((px_r_slice + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0])
    #                 print("a: ", ((px_a_slice + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1])

    #             elif angle_bin >= (self.OUTPUT_DIM[2] - s) and range_bin < s:
    #                 end = s + (self.OUTPUT_DIM[2] - angle_bin)
    #                 range_slice = range_bin + s + 1
    #                 px_r_slice = px_r[s - range_bin:, :end]
    #                 px_a_slice = px_a[s - range_bin:, :end]
    
    #                 map[0, 0:range_slice, angle_start:] = 1
    #                 map[1, 0:range_slice, angle_start:] = ((px_r_slice + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
    #                 map[2, 0:range_slice, angle_start:] = ((px_a_slice + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
    #                 print("range_slice: ", range_slice)
    #                 print("angle_start: ", angle_start)
    #                 print("r: ", ((px_r_slice + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0])
    #                 print("a: ", ((px_a_slice + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1])

    #             elif angle_bin >= s and range_bin >= (self.OUTPUT_DIM[1] - s):
    #                 end_range = s + (self.OUTPUT_DIM[1] - range_bin)
    #                 px_r_slice = px_r[:end_range, :]
    #                 px_a_slice = px_a[:end_range, :]
    
    #                 map[0, range_start:, angle_start:angle_end] = 1
    #                 map[1, range_start:, angle_start:angle_end] = ((px_r_slice + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
    #                 map[2, range_start:, angle_start:angle_end] = ((px_a_slice + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
    #                 print("range_start: ", range_start)
    #                 print("angle_start:angle_end: ", angle_start, angle_end)
    #                 print("r: ", ((px_r_slice + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0])
    #                 print("a: ", ((px_a_slice + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1])

    #             elif angle_bin < s and range_bin >= (self.OUTPUT_DIM[1] - s):
    #                 angle_slice = angle_bin + s + 1
    #                 end = s + (self.OUTPUT_DIM[1] - range_bin)
    #                 px_r_slice = px_r[:end, s - angle_bin:]
    #                 px_a_slice = px_a[:end, s - angle_bin:]
    
    #                 map[0, range_start:, 0:angle_slice] = 1
    #                 map[1, range_start:, 0:angle_slice] = ((px_r_slice + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
    #                 map[2, range_start:, 0:angle_slice] = ((px_a_slice + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
    #                 print("range_start: ", range_start)
    #                 print("angle_slice: ", angle_slice)
    #                 print("r: ", ((px_r_slice + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0])
    #                 print("a: ", ((px_a_slice + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1])

    #             elif angle_bin >= (self.OUTPUT_DIM[2] - s) and range_bin >= (self.OUTPUT_DIM[1] - s):
    #                 end_angle = s + (self.OUTPUT_DIM[2] - angle_bin)
    #                 end_range = s + (self.OUTPUT_DIM[1] - range_bin)
    #                 px_r_slice = px_r[:end_range, :end_angle]
    #                 px_a_slice = px_a[:end_range, :end_angle]
    
    #                 map[0, range_start:, angle_start:] = 1
    #                 map[1, range_start:, angle_start:] = ((px_r_slice + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
    #                 map[2, range_start:, angle_start:] = ((px_a_slice + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
    #                 print("range_start: ", range_start)
    #                 print("angle_start: ", angle_start)
    #                 print("r: ", ((px_r_slice + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0])
    #                 print("a: ", ((px_a_slice + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1])

    #             else:
    #                 print("none of above fits")
    #                 # Stop the script
    #                 sys.exit("Stopping the script")  

    #             # if(angle_bin>=s and angle_bin<(self.OUTPUT_DIM[2]-s)):

    #             #     map[0,range_bin-s:range_bin+(s+1),angle_bin-s:angle_bin+(s+1)] = 1
    #             #     map[1,range_bin-s:range_bin+(s+1),angle_bin-s:angle_bin+(s+1)] = ((px_r+range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
    #             #     map[2,range_bin-s:range_bin+(s+1),angle_bin-s:angle_bin+(s+1)] = ((px_a + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1] 
    #             # elif(angle_bin<s):
    #             #     map[0,range_bin-s:range_bin+(s+1),0:angle_bin+(s+1)] = 1
    #             #     map[1,range_bin-s:range_bin+(s+1),0:angle_bin+(s+1)] = ((px_r[:,s-angle_bin:] + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
    #             #     map[2,range_bin-s:range_bin+(s+1),0:angle_bin+(s+1)] = ((px_a[:,s-angle_bin:] + angle_mod)- self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
                    
    #             # elif(angle_bin>=self.OUTPUT_DIM[2]):
    #             #     end = s+(self.OUTPUT_DIM[2]-angle_bin)
    #             #     map[0,range_bin-s:range_bin+(s+1),angle_bin-s:] = 1
    #             #     map[1,range_bin-s:range_bin+(s+1),angle_bin-s:] = ((px_r[:,:end] + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
    #             #     map[2,range_bin-s:range_bin+(s+1),angle_bin-s:] = ((px_a[:,:end] + angle_mod)- self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]

    #     return map

    def rd_encode(self,labels):
        selected_dims = (self.OUTPUT_DIM[0], self.OUTPUT_DIM[1], self.OUTPUT_DIM[3])
        #print("selected_dims: ", selected_dims)

        # Initialize the numpy array with the selected dimensions
        map = np.zeros(selected_dims)
        #map = np.zeros(self.OUTPUT_DIM )

        #print("encoder map shape: ", map.shape)

        for lab in labels:
            # [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix]

            if(lab[0]==-1):
                continue

            range_bin = int(np.clip(lab[0]/self.geometry['resolution'][0]/4,0,self.OUTPUT_DIM[1]-1))
            range_mod = lab[0] - range_bin*self.geometry['resolution'][0]*4

            doppler_bin = int(np.clip(np.floor(lab[2]/self.geometry['resolution'][2] + self.OUTPUT_DIM[3]/2),0,self.OUTPUT_DIM[3]-1))
            doppler_mod = lab[1] - (doppler_bin- self.OUTPUT_DIM[2]/2)*self.geometry['resolution'][2]

            
            if(self.geometry['size']==1):
                
                map[0,range_bin,doppler_bin] = 1
                map[1,range_bin,doppler_bin] = (range_mod - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                map[2,range_bin,doppler_bin] = (doppler_mod - self.statistics['reg_mean'][2]) / self.statistics['reg_std'][2]

            else:
                s = int((self.geometry['size']-1)/2) #size=3 (3-1)/2=1
                #print("s: ", s)
                r_lin = np.linspace(self.geometry['resolution'][0]*s, -self.geometry['resolution'][0]*s,
                                    self.geometry['size'])*4
                a_lin = np.linspace(self.geometry['resolution'][2]*s, -self.geometry['resolution'][2]*s,
                                    self.geometry['size'])

                px_a, px_r = np.meshgrid(a_lin, r_lin)
                #print("******************rd_encoder")
                #print("self.OUTPUT_DIM: ", self.OUTPUT_DIM)
                #print("self.OUTPUT_DIM[0, 1, 3]: ", selected_dims)

                map = update_map(map, range_bin, doppler_bin, s, px_r, px_a, range_mod, doppler_mod, self.statistics, selected_dims)

                
        return map
    
    def ra_encode(self,labels):
        selected_dims = (self.OUTPUT_DIM[0], self.OUTPUT_DIM[1], self.OUTPUT_DIM[2])

        # Initialize the numpy array with the selected dimensions
        map = np.zeros(selected_dims)
        #map = np.zeros(self.OUTPUT_DIM )

        #print("encoder map shape: ", map.shape)

        for lab in labels:
            # [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix]

            if(lab[0]==-1):
                continue

            range_bin = int(np.clip(lab[0]/self.geometry['resolution'][0]/4,0,self.OUTPUT_DIM[1]-1))
            range_mod = lab[0] - range_bin*self.geometry['resolution'][0]*4

            # ANgle and deg
            angle_bin = int(np.clip(np.floor(lab[1]/self.geometry['resolution'][1]/4 + self.OUTPUT_DIM[2]/2),0,self.OUTPUT_DIM[2]-1))
            angle_mod = lab[1] - (angle_bin- self.OUTPUT_DIM[2]/2)*self.geometry['resolution'][1]*4 

            
            if(self.geometry['size']==1):
                
                map[0,range_bin,angle_bin] = 1
                map[1,range_bin,angle_bin] = (range_mod - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                map[2,range_bin,angle_bin] = (angle_mod - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]

            else:
                s = int((self.geometry['size']-1)/2) #size=3 (3-1)/2=1
                #print("s: ", s)
                r_lin = np.linspace(self.geometry['resolution'][0]*s, -self.geometry['resolution'][0]*s,
                                    self.geometry['size'])*4
                a_lin = np.linspace(self.geometry['resolution'][1]*s, -self.geometry['resolution'][1]*s,
                                    self.geometry['size'])*4

                px_a, px_r = np.meshgrid(a_lin, r_lin)

                #print("******************ra_encoder")
                #print("self.OUTPUT_DIM: ", self.OUTPUT_DIM)
                #print("self.OUTPUT_DIM[0, 1, 3]: ", selected_dims)

                map = update_map(map, range_bin, angle_bin, s, px_r, px_a, range_mod, angle_mod, self.statistics, selected_dims)

                
        return map

    def decode(self,ra_map,rd_map,threshold):
       
        range_bins,angle_bins = np.where(ra_map[0,:,:]>=threshold)

        coordinates = []

        for range_bin,angle_bin in zip(range_bins,angle_bins):
            R = range_bin*4*self.geometry['resolution'][0] + ra_map[1,range_bin,angle_bin] * self.statistics['reg_std'][0] + self.statistics['reg_mean'][0]
            A = (angle_bin-self.OUTPUT_DIM[2]/2)*4*self.geometry['resolution'][1] + ra_map[2,range_bin,angle_bin] * self.statistics['reg_std'][1] + self.statistics['reg_mean'][1]
            C = ra_map[0,range_bin,angle_bin]
            # Get Doppler from the RD map using the same range bin
            doppler_bins = np.argmax(rd_map[0, range_bin, :])  # Max Doppler bin for this range bin

            # Decode Doppler from the RD map
            D = (doppler_bins-self.OUTPUT_DIM[3]/2)*self.geometry['resolution'][2] + rd_map[1, range_bin, doppler_bins] * self.statistics['reg_std'][2] + self.statistics['reg_mean'][2]
            #  map[1,range_bin,angle_bin]: Model prediction on range, reg head
            #  map[2,range_bin,angle_bin]: Model prediction on angle, reg head
        
            coordinates.append([R,A,D,C])
       
        return coordinates

###### functions for encoder
# Function to calculate start and end indices
def get_start_end_indices(bin_idx, s, output_dim):
    start = max(bin_idx - s, 0)
    end = min(bin_idx + s + 1, output_dim)
    return start, end

# Function to get sliced px data
def get_sliced_data(bin_idx, s, px_r, px_a, output_dim, dim_type):
    if dim_type == "range":
        if bin_idx >= s and bin_idx < (output_dim - s):
            return px_r, px_a#, px_v  # Central case, no slicing needed
        elif bin_idx < s:
            return px_r[s - bin_idx:, :], px_a[s - bin_idx:, :]#, px_v[s - bin_idx:, :]  # Left boundary case
        else:
            return px_r[:s + (output_dim - bin_idx), :], px_a[:s + (output_dim - bin_idx), :]#, px_v[:s + (output_dim - bin_idx), :]  # Right boundary case
    elif dim_type == "other":
        if bin_idx >= s and bin_idx < (output_dim - s):
            return px_r, px_a#, px_v  # Central case, no slicing needed
        elif bin_idx < s:
            return px_r[:, s - bin_idx:], px_a[:, s - bin_idx:]#, px_v[:, s - bin_idx:]  # Left boundary case
        else:
            return px_r[:, :s + (output_dim - bin_idx)], px_a[:, :s + (output_dim - bin_idx)]#, px_v[:, :s + (output_dim - bin_idx)]  # Right boundary case
    # elif dim_type == "velocity":
    #     if bin_idx >= s and bin_idx < (output_dim - s):
    #         return px_r, px_a, px_v  # Central case, no slicing needed
    #     elif bin_idx < s:
    #         return px_r[:, :, s - bin_idx:], px_a[:, :, s - bin_idx:], px_v[:, :, s - bin_idx:]  # Left boundary case
    #     else:
    #         return px_r[:, :, :s + (output_dim - bin_idx)], px_a[:, :, :s + (output_dim - bin_idx)], px_v[:, :, :s + (output_dim - bin_idx)]  # Right boundary case

# Function to update the map based on provided indices and slices
def update_map_slice(map, indices, px_r_slice, px_a_slice, range_mod, angle_mod, reg_mean, reg_std):
    range_start, range_end, angle_start, angle_end = indices

    map[0, range_start:range_end, angle_start:angle_end] = 1
    map[1, range_start:range_end, angle_start:angle_end] = (px_r_slice + range_mod - reg_mean[0]) / reg_std[0]
    map[2, range_start:range_end, angle_start:angle_end] = (px_a_slice + angle_mod - reg_mean[1]) / reg_std[1]
    #print("range_start:range_end: ", range_start, range_end)
    #print("angle_start:angle_end: ", angle_start, angle_end)
    #print("r: ", (px_r_slice + range_mod - reg_mean[0]) / reg_std[0])
    #print("a: ", (px_a_slice + angle_mod - reg_mean[1]) / reg_std[1])
    #map[3, range_start:range_end, angle_start:angle_end, vel_start:vel_end] = (px_v_slice + vel_mod - reg_mean[2]) / reg_std[2]

    return map

# Function to handle each case for range, angle, and velocity bins
def process_map_case(map, range_bin, angle_bin, s, px_r, px_a, range_mod, angle_mod, reg_mean, reg_std, output_dim):
    # Get start and end indices for range, angle, and velocity
    range_start, range_end = get_start_end_indices(range_bin, s, output_dim[1])
    angle_start, angle_end = get_start_end_indices(angle_bin, s, output_dim[2])
    #vel_start, vel_end = get_start_end_indices(vel_bin, s, output_dim[3])

    # Slice the px_r, px_a, and px_v data based on range, angle, and velocity bin cases
    px_rr, px_ar = get_sliced_data(range_bin, s, px_r, px_a, output_dim[1], 'range')
    #px_ra, px_aa = get_sliced_data(angle_bin, s, px_rr, px_ar, output_dim[2], 'angle')
    px_r_slice, px_a_slice = get_sliced_data(angle_bin, s, px_rr, px_ar, output_dim[2], 'other')
    #px_r_slice, px_a_slice = get_sliced_data(vel_bin, s, px_ra, px_aa, px_va, output_dim[3], 'velocity')

    # Update the map for the current slice
    map = update_map_slice(map, (range_start, range_end, angle_start, angle_end),
                     px_r_slice, px_a_slice, range_mod, angle_mod, reg_mean, reg_std)
    
    return map

# Main function that handles map updates
def update_map(map, range_bin, angle_bin, s, px_r, px_a, range_mod, angle_mod, statistics, output_dim):
    reg_mean = statistics['reg_mean']
    reg_std = statistics['reg_std']

    # # Get case indicators for range, angle, and velocity bins
    # range_case = 0 if s <= range_bin < (output_dim[1] - s) else (-1 if range_bin < s else 1)
    # angle_case = 0 if s <= angle_bin < (output_dim[2] - s) else (-1 if angle_bin < s else 1)
    # vel_case = 0 if s <= vel_bin < (output_dim[3] - s) else (-1 if vel_bin < s else 1)

    # Process map for all cases
    map = process_map_case(map, range_bin, angle_bin, s, px_r, px_a,
                     range_mod, angle_mod, reg_mean, reg_std, output_dim)
    
    return map