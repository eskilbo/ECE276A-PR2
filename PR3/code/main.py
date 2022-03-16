import numpy as np
from pr3_utils import *
from tqdm import tqdm
from scipy.linalg import inv
from mapping import EKFMapping
from slam import EKFSlam

if __name__ == '__main__':

	# Load the measurements
	filename = "./data/10.npz"
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
	
	features_downsampled = features[:, 0:features.shape[1]:50, :]
	ekf_mapping = EKFMapping(n_features=features_downsampled.shape[1],iTcam=imu_T_cam,K_=K,baseline=b)
	ekf_slam = EKFSlam(n_features=features_downsampled.shape[1],iTcam=imu_T_cam,K_=K,baseline=b)

	u = np.vstack([linear_velocity,angular_velocity])
	T = linear_velocity.shape[-1]

	# Initial imu trajectory
	trajectory_imu_a = np.zeros((4,4,T))
	trajectory_imu_a[:,:,0] = np.eye(4)

	trajectory_imu_c = np.zeros((4,4,T))
	trajectory_imu_c[:,:,0] = np.eye(4)

	for i in tqdm(range(1, T),desc="Progress"):
		# (a) IMU Localization via EKF Prediction
		ekf_mapping.predict(u[:,i], (t[0, i] - t[0, i-1]))
		trajectory_imu_a[:,:,i] = inv(ekf_mapping.mu_imu)
		# (b) Landmark Mapping via EKF Update
		ekf_mapping.update(features_downsampled[:, :, i])
		# (c) Visual-Inertial SLAM
		ekf_slam.predict(u[:,i], (t[0, i] - t[0, i-1]))
		trajectory_imu_c[:,:,i] = inv(ekf_slam.mu_imu)
		zmap = ekf_slam.update(features_downsampled[:, :, i])
	# You can use the function below to visualize the robot pose over time
	visualize_trajectory_2d(trajectory_imu_a,landmarks=ekf_mapping.mu_landmark, path_name=None, show_ori = True)
	visualize_trajectory_2d(trajectory_imu_c,landmarks=ekf_slam.mu_landmark, path_name=None, show_ori = True)
	#visualize_trajectory_2d(trajectory_imu,landmarks=None, path_name=None, show_ori = True)
