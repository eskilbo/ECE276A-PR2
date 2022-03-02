from pr2_utils import *
import numpy as np
import matplotlib.pyplot as plt; plt.ion()

def update_map(MAP, particle_max_state, ranges, angles, lTb):

    # Body2World transform
    x_world = particle_max_state[0]
    y_world = particle_max_state[1]
    theta_world = particle_max_state[2]
    bTw = np.array([[np.cos(theta_world), -np.sin(theta_world), 0, x_world],
                    [np.sin(theta_world),  np.cos(theta_world), 0, y_world],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # Start point of lidar ray in lidar frame
    sx = np.ceil((x_world - MAP['xmin']) / MAP['res']).astype(np.int16)-1
    sy = np.ceil((y_world - MAP['ymin']) / MAP['res']).astype(np.int16)-1

    # End point of lidar ray in lidar frame
    ex = ranges * np.cos(angles)
    ey = ranges * np.sin(angles)

    # End points
    exyz = np.ones((4, np.size(ex)))
    exyz[0, :] = ex
    exyz[1, :] = ey

    # Transform end point to world frame
    exyz = np.dot(bTw, np.dot(lTb, exyz))

    # Convert end point to cells
    ex = exyz[0, :]
    ey = exyz[1, :]
    ex = np.ceil((ex - MAP['xmin']) / MAP['res']).astype(np.int16)-1
    ey = np.ceil((ey - MAP['ymin']) / MAP['res']).astype(np.int16)-1

    for i in range(np.size(ranges)):
        bresenham_points = bresenham2D(sx, sy, ex[i], ey[i])
        bresenham_points_x = bresenham_points[0, :].astype(np.int16)
        bresenham_points_y = bresenham_points[1, :].astype(np.int16)

        indGood = np.logical_and(
            np.logical_and(np.logical_and((bresenham_points_x > 1), (bresenham_points_y > 1)), (bresenham_points_x < MAP['sizex'])), (bresenham_points_y < MAP['sizey']))

        # Decrease log-odds if cell is free
        MAP['map'][bresenham_points_x[indGood], bresenham_points_y[indGood]] -= np.log(4)

        # Increase log-odds if cell is occupied
        if ((ex[i] > 1) and (ex[i] < MAP['sizex']) and (ey[i] > 1) and (ey[i] < MAP['sizey'])):
            MAP['map'][ex[i], ey[i]] += 2*np.log(4)

    # clip range to prevent over-confidence
    MAP['map'] = np.clip(MAP['map'], -10*np.log(4), 10*np.log(4))
    return MAP