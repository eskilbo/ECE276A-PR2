import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import matplotlib
matplotlib.use('TkAgg')
from pr2_utils import init_map,init_particles,lidar_transforms,fog_transforms,lidar,encoder
from mapping import update_map
from particle_filter import prediction, update, resampling, linear_v, lidar_ranges_angles
from tqdm import tqdm

# Lidar
lidar_size,lidar_time,lidar_data,start_angle,end_angle = lidar()

# Fog angular velocity
w = np.load('data/w_new.npy')

# Encoder
encoder_size,encoder_time,encoder_data,enc_left_count,enc_right_count,enc_resolution,d_enc_left,d_enc_right = encoder()

# Store transforms from vehicle2fog,lidar,stereo from param folder
RPY_Fog,R_Fog,T_Fog = fog_transforms()
RPY_Lidar,R_Lidar,T_Lidar = lidar_transforms()
lTb = np.array([[R_Lidar[0][0],R_Lidar[0][1],R_Lidar[0][2],T_Lidar[0]],
                [R_Lidar[1][0],R_Lidar[1][1],R_Lidar[1][2],T_Lidar[1]],
                [R_Lidar[2][0],R_Lidar[2][1],R_Lidar[2][2],T_Lidar[2]],
                [0, 0, 0, 1]])

# Initialize map
MAP = init_map()

# Initialize particles
N = 40
particle_state, particle_weight = init_particles(N)

# Convert angles to angle and lidar range values
ranges, angles = lidar_ranges_angles(lidar_data,0,start_angle,end_angle)

# Update map
particle_max_position = np.argmax(particle_weight)
particle_max_state = particle_state[:, particle_max_position]
MAP = update_map(MAP, particle_max_state, ranges, angles, lTb)
plt.imshow(MAP['map'], cmap='gray')

trajectory = np.array([[0],[0]])
encoder_idx = 1
lidar_idx = 0

for i in tqdm(range(0, encoder_size+lidar_size),desc="Progress"):
    if i % 2000 == 0:
        # Recover map pmf from log-odds map
        output_map = ((1 - 1 / (1 + np.exp(MAP['map']))) < 0.1).astype(int)

        # Convert end point in world frame to cells
        ex = np.ceil((trajectory[0, :] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
        ey = np.ceil((trajectory[1, :] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

        indGood = np.logical_and(np.logical_and(np.logical_and((ex > 1), (ey > 1)), (ex < MAP['sizex'])),
                                 (ey < MAP['sizey']))

        output_map[ex[indGood], ey[indGood]] = 2

        # Plot map every 1000th iteration
        plt.imshow(MAP['map'], cmap='hot')
        plt.title("Map")
        plt.pause(0.001)

        # Save map every 10000th iteration
        if i % 10000 == 0:
            plt.imsave('plots/' + str(i) + '.png', output_map, cmap='gray')
    
    # Recover last map before iterations stop
    if i == encoder_size+lidar_size-10:
        # Recover map pmf from log-odds map
        output_map = ((1 - 1 / (1 + np.exp(MAP['map']))) < 0.1).astype(int)

        # Convert end point in world frame to cells
        ex = np.ceil((trajectory[0, :] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
        ey = np.ceil((trajectory[1, :] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

        indGood = np.logical_and(np.logical_and(np.logical_and((ex > 1), (ey > 1)), (ex < MAP['sizex'])),
                                 (ey < MAP['sizey']))

        output_map[ex[indGood], ey[indGood]] = 2
        plt.imsave('plots/' + 'last_map' + '.png', output_map, cmap='gray')
        break

    if encoder_time[encoder_idx] < lidar_time[lidar_idx]:
        tau = (encoder_time[encoder_idx] - encoder_time[encoder_idx-1]) / 10**9
        linear_velocity = linear_v(tau,encoder_idx,d_enc_left,enc_left_count,d_enc_right,enc_right_count,enc_resolution)
        particle_state = prediction(particle_state, tau, linear_velocity, w[encoder_idx])
        if encoder_idx < encoder_size-1:
            encoder_idx += 1
        else:
            lidar_idx += 1
    else:
        ranges, angles = lidar_ranges_angles(lidar_data,lidar_idx,start_angle,end_angle)
        particle_state, particle_weight = update(MAP, particle_state, particle_weight, ranges, angles, lTb, N)
        particle_max_position = np.argmax(particle_weight)
        particle_max_state = particle_state[:,particle_max_position]
        trajectory = np.hstack((trajectory,particle_max_state[0:2].reshape(2,1)))
        MAP = update_map(MAP, particle_max_state, ranges, angles, lTb)
        # Resample the particles 
        N_eff = 1/np.dot(particle_weight.reshape(1,N), particle_weight.reshape(N,1))
        if N_eff < 20:
            particle_state, particle_weight = resampling(particle_state, particle_weight, N)
        if (lidar_idx < lidar_size):
            lidar_idx += 1
        else:
            encoder_idx += 1