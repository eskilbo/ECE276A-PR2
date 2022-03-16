import numpy as np
from pr2_utils import mapCorrelation

def prediction(particle_state, tau, v_t, w_t):
    N = np.shape(particle_state)[1]
    x_w = particle_state[0,:]
    y_w = particle_state[1,:]
    theta_w = particle_state[2,:]
    delta_x = tau * v_t * np.cos(theta_w)
    delta_y = tau * v_t * np.sin(theta_w)
    x_w += delta_x
    y_w += delta_y
    theta_w += tau * w_t

    # Add noise 
    noise_x = np.random.normal(0,abs(np.max(delta_x))/10,N)
    noise_y = np.random.normal(0,abs(np.max(delta_y))/10,N)
    noise_theta = np.random.normal(0,abs(tau*w_t)/10,N)

    particle_state[0,:] = x_w + noise_x
    particle_state[1,:] = y_w + noise_y
    particle_state[2,:] = theta_w + noise_theta
    return particle_state

def update(MAP, particle_state, particle_weight, ranges, angles, lTb, N):

    # Binary map for map correlation
    map = ((1 - 1 / (1 + np.exp(MAP['map']))) > 0.5).astype(np.int)

    x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  # x index of each pixel on log-odds map
    y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  # y index of each pixel on log-odds map

    # Grid around particle
    x_range = np.arange(-4 * MAP['res'], 5 * MAP['res'], MAP['res'])  # x deviation
    y_range = np.arange(-4 * MAP['res'], 5 * MAP['res'], MAP['res'])  # y deviation

    # Point of lidar ray in lidar frame (xy)
    ex = ranges * np.cos(angles)
    ey = ranges * np.sin(angles)

    # Convert to 2D (xy)
    exy = np.ones((4, np.size(ex)))
    exy[0, :] = ex
    exy[1, :] = ey
    exy[3, :] = 0

    # Transform to body frame
    exy = np.dot(lTb, exy)

    correlation = np.zeros(N)

    for i in range(N):
        x_t = particle_state[:, i]
        x_w = x_t[0]
        y_w = x_t[1]
        theta_w = x_t[2]

        # Transform to world frame
        bTw = np.array([[np.cos(theta_w), -np.sin(theta_w), 0, x_w],
                        [np.sin(theta_w), np.cos(theta_w), 0, y_w],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        s_w = np.dot(bTw, exy)
        ex_w = s_w[0, :]
        ey_w = s_w[1, :]
        Y = np.stack((ex_w, ey_w))

        c = mapCorrelation(map, x_im, y_im, Y, x_range, y_range)

        correlation[i] = np.max(c)

    # Update particle weight with softmax
    p_h = softmax(correlation)
    particle_weight *= p_h / np.sum(particle_weight * p_h)
    return particle_state, particle_weight

def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / np.sum(e_x)

def resampling(particle_state, particle_weight, N):

    particle_state_new = np.zeros((3, N))
    particle_weight_new = np.tile(1 / N, N).reshape(1, N)
    j = 0
    c = particle_weight[0, 0]

    for i in range(N):
        u = np.random.uniform(0, 1/N)
        b = u + i/N
        while b > c:
            j += 1
            c += particle_weight[0, j]
        particle_state_new[:, i] = particle_state[:, j]

    return particle_state_new, particle_weight_new

def linear_v(tau,encoder_idx,d_enc_left,enc_left_count,d_enc_right,enc_right_count,resolution):
    if encoder_idx != 0:
        left_linear_velocity = (np.pi * d_enc_left * (enc_left_count[encoder_idx] - enc_left_count[encoder_idx-1])) / (resolution * tau)
    else:
        left_linear_velocity = (np.pi * d_enc_left * enc_left_count[encoder_idx]) / resolution
    if encoder_idx != 0:
        right_linear_velocity = (np.pi * d_enc_right * (enc_right_count[encoder_idx] - enc_right_count[encoder_idx-1])) / (resolution * tau)
    else:
        right_linear_velocity = (np.pi * d_enc_right * enc_right_count[encoder_idx]) / resolution
    return (left_linear_velocity + right_linear_velocity) / 2

def lidar_ranges_angles(lidar_data,lidar_idx,start_angle,end_angle):
    angles = np.linspace(start_angle, end_angle, 286) / 180 * np.pi
    ranges = lidar_data[lidar_idx, :]
    indValid = np.logical_and((ranges < 55),(ranges > 1.5))
    ranges = ranges[indValid]
    angles = angles[indValid]
    return ranges, angles