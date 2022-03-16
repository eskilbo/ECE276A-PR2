from turtle import update
from cv2 import exp
import numpy as np
from scipy.linalg import expm, inv
from pr3_utils import *

class EKFMapping:
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
        self.sigma_landmark = np.identity(3*n_features) * self.V
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

    def get_H(self, zmap):
        n_obs = zmap.size
        n_updates = self.init_max_idx
        P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        mu_landmark = np.hstack([self.mu_landmark[zmap,:],np.ones((n_obs,1))])
        H = np.zeros((n_obs * 4, n_updates * 3))
        for i in range(n_obs):
            m_i = zmap[i]
            wTc = self.iTc@self.mu_imu
            dpidq = dpi_dq(wTc@mu_landmark[i,:].reshape(-1,1))
            H[4*i:4*(i+1), 3*m_i:3*(m_i+1)] = self.M @ dpidq @ wTc @ P.T
        return H
    
    def get_jacobian(self, update_feature_index, prior):
        cTw = self.iTc @ self.mu_imu
        H = np.zeros((4*np.size(update_feature_index),3*self.n_features))
        for i in range(np.size(update_feature_index)):
            current_index = update_feature_index[i]
            H[i*4:(i+1)*4,current_index*3:(current_index+1)*3] =  np.dot(np.dot(np.dot(self.M,dpi_dq(np.dot(cTw,prior[:,i]))),cTw),self.P.T)
        return H
        
    def update(self, z):
        sum_of_feature_vectors = np.sum(z[:, :], 0)
        obs_feats_idx = np.array(np.where(sum_of_feature_vectors != -4))
        update_feats_idx = np.zeros((0, 0), dtype=np.int8)
        update_feats = np.zeros((4,0))
        if obs_feats_idx.size > 0:
            obs_features = self.init_features(z,obs_feats_idx)
            for i in range(obs_feats_idx.size):
                m_i = obs_feats_idx[0,i]
                if (np.array_equal(self.mu_landmark[:,m_i],[-1,-1,-1,-1])):
                    self.mu_landmark[:,m_i] = obs_features[:,i]
                else:
                    update_feats_idx = np.append(update_feats_idx,m_i)
                    update_feats = np.hstack((update_feats,obs_features[:,i].reshape(4,1)))

            if update_feats_idx.size != 0:
                mu_landmark = self.mu_landmark[:, update_feats_idx]
                mu_landmark.reshape((4, update_feats_idx.size))
                H = self.get_jacobian(update_feats_idx,mu_landmark)
                K = (self.sigma_landmark @ np.transpose(H)) @ inv(((H @ self.sigma_landmark)@np.transpose(H))+np.identity(4*update_feats_idx.size)*self.V)
                z_ = z[:,update_feats_idx].reshape((4,update_feats_idx.size))
                z_hat = self.M @ pi((self.iTc@self.mu_imu) @ mu_landmark)
                self.mu_landmark = (self.mu_landmark.reshape(-1,1,order='F') + ((self.D @ K) @ (z_-z_hat).reshape(-1,1,order='F'))).reshape(4,-1,order='F')
                self.sigma_landmark = (np.identity(3*self.n_features)-(K @ H)) @ self.sigma_landmark
    