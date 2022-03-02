import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import matplotlib
matplotlib.use('TkAgg')
from pr2_utils import read_data_from_csv
from particle_filter import prediction
from tqdm import tqdm

fog_time, fog_data = read_data_from_csv('data/sensor_data/fog.csv')
w = np.load('data/w_new.npy')

encoder_time, encoder_data = read_data_from_csv('data/sensor_data/encoder.csv')
encoder_left_count = encoder_data[:, 0]
encoder_right_count = encoder_data[:, 1]
encoder_resolution = 4096
encoder_left_wheel_diameter = 0.623479
encoder_right_wheel_diameter = 0.622806

# Initialize particles
N = 1
particle_state = np.zeros((3,N))
particle_weight = np.zeros((1,N))
particle_weight[0, 0:N] = 1 / N

x = []
y = []
for i in tqdm(range(1, len(encoder_time))):
    tau = (encoder_time[i]-encoder_time[i-1])/1e+9
    left_linear_velocity = (np.pi * encoder_left_wheel_diameter * (encoder_left_count[i] - encoder_left_count[i-1])) / (encoder_resolution * tau)
    right_linear_velocity = (np.pi * encoder_right_wheel_diameter * (encoder_right_count[i] - encoder_right_count[i-1])) / (encoder_resolution * tau)
    linear_velocity = (left_linear_velocity + right_linear_velocity) / 2
    particle_state = prediction(particle_state,tau,linear_velocity,w[i])
    x.append(particle_state[0][0])
    y.append(particle_state[1][0])

plt.plot(x,y)
plt.title('Dead Reckoning')
plt.savefig('plots/deadreckoning.png')
plt.show()
plt.pause(1)