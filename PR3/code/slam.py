from cv2 import exp
import numpy as np
from scipy.linalg import expm, inv
from pr3_utils import *

class EKFSlam:
    def __init__(self,
                n_features,
                iTcam,
                K_,
                baseline):
        self.mu_imu = np.eye(4)
        self.sigma_imu = np.eye(6)
        self.V = 100
        self.W = 1e-3 * np.eye(6)
        self.n_features = n_features
        self.mu_landmark = -1 * np.ones((4,n_features))
        self.D = np.kron(np.eye(n_features),np.vstack((np.identity(3),np.zeros((1,3)))))
        self.P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        self.iTc = inv(iTcam)
        K = K_
        self.b = baseline
        self.M = np.array([[K[0][0], 0, K[0][2], 0],
                  [0, K[1][1], K[1][2], 0],
                  [K[0][0], 0, K[0][2], -K[0][0] * self.b],
                  [0, K[1][1], K[1][2], 0]])
    
    def predict(self, u, tau):
        self.mu_imu = expm(-tau*hat_adj(u))@self.mu_imu
        self.sigma_imu = np.dot(np.dot(expm(-tau*cwedge(u)),self.sigma_imu),np.transpose(expm(-tau*cwedge(u)))) + self.W
    
    def init_features(self, z, obs_feats_idx):
        num_obs_feats = obs_feats_idx.size
        if num_obs_feats > 0:
            wTc = inv(self.iTc @ self.mu_imu)
            feats_coords = z[:,obs_feats_idx].reshape(4,num_obs_feats)
            obs_features = np.ones((4, num_obs_feats))
            obs_features[0, :] = (feats_coords[0, :] - self.M[0, 2]) * self.b / (feats_coords[0, :] - feats_coords[2, :])
            obs_features[1, :] = (feats_coords[1, :] - self.M[1, 2]) * (-self.M[2, 3]) / (self.M[1, 1] * (feats_coords[0, :] - feats_coords[2, :]))
            obs_features[2, :] = -self.M[2, 3] / (feats_coords[0, :] - feats_coords[2, :])
            obs_features = wTc @ obs_features
        return obs_features
        
    def update(self, z):
        sum_of_feature_vectors = np.sum(z[:, :], 0)
        obs_feats_idx = np.array(np.where(sum_of_feature_vectors != -4))
        H = np.zeros((4*obs_feats_idx.size,6))
        z_est = np.zeros(4*obs_feats_idx.size)
        z_obs = np.zeros(4*obs_feats_idx.size)
        j = 0
        if obs_feats_idx.size > 0:
            obs_features = self.init_features(z,obs_feats_idx)
            for i in range(obs_feats_idx.size):
                m_i = obs_feats_idx[0,i]
                if (np.array_equal(self.mu_landmark[:,m_i],[-1,-1,-1,-1])):
                    self.mu_landmark[:,m_i] = obs_features[:,i]
                else:
                    mi_h = np.block([self.mu_landmark[:, i],1])
                    mi_h = self.mu_landmark[:,i]
                    z_obs[4*j:4*j+4] = z[:,i]
                    z_est[4*j:4*j+4] = (self.M @ pi_2((self.iTc@inv(self.mu_imu))@mi_h)).ravel()
                    H[4*j:4*j+4,:] = -self.M @ dpi_dq((self.iTc@inv(self.mu_imu))@mi_h) @ self.iTc @ cdot(inv(self.mu_imu)@mi_h)
        
            K = (self.sigma_imu @ np.transpose(H)) @ inv(((H @ self.sigma_imu)@np.transpose(H))+np.identity(4*obs_feats_idx.size)*self.V)
            mu_update = K @ (z_obs.T - z_est)
            skew = hat_adj(mu_update[3:])
            u = np.block([[skew,mu_update[3:].reshape(-1,1)],[np.zeros(3).reshape(-1,1).T,0]])
            self.mu_imu = self.mu_imu@expm(u)
            self.sigma_imu = (np.eye(6)-K@H)@self.sigma_imu