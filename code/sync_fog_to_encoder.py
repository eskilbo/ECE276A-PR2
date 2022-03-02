from pr2_utils import read_data_from_csv
import numpy as np

lidar_time, lidar_data = read_data_from_csv('data/sensor_data/lidar.csv')
fog_time, fog_data = read_data_from_csv('data/sensor_data/fog.csv')
fog_delta_yaw = fog_data[:, 2]
encoder_time, encoder_data = read_data_from_csv('data/sensor_data/encoder.csv')

lidar_length = len(lidar_time)
fog_length = len(fog_time)
encoder_length = len(encoder_time)

tau = 1e-9 * fog_time[:1160502]
fog_delta_yaw = fog_data[:, 2]
fog_delta_yaw_synced = []
fog_time_synced = []
for i in range(10 , len(fog_delta_yaw[:1160501]), 10): 
    fog_time_synced.append(tau[i] - tau[i - 10])
    fog_delta_yaw_synced.append(np.sum(fog_delta_yaw[i - 10 : i]))
fog_delta_yaw_synced = np.asarray(fog_delta_yaw_synced)  
fog_time_synced = np.asarray(fog_time_synced)
np.save('data/w_new.npy', fog_delta_yaw_synced/fog_time_synced)