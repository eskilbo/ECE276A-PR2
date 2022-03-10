import os, cv2
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np

def create_blue_data():
    pixels = []
    labels = []
    folder = 'data/training'
    for i, filename in enumerate(os.listdir(folder)):
        if i == 30:
            break
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder,filename)) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # display the image and use roipoly for labeling
            fig, ax = plt.subplots()
            ax.imshow(img)
            my_roi = RoiPoly(fig=fig, ax=ax, color='r')
            
            # get the image mask
            mask = my_roi.get_mask(img)

            for y in range(mask.shape[0]):
                for x in range(mask.shape[1]):
                    if mask[y][x]:
                        pixels.append(img[y][x])
                        labels.append(int(1))
            
            # display the labeled region and the image mask
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('%d pixels selected\n' % img[mask,:].shape[0])
            
            ax1.imshow(img)
            ax1.add_line(plt.Line2D(my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))
            ax2.imshow(mask)
            plt.show(block=True)
    return np.array(pixels), np.array(labels)

def create_other_data():
    pixels = []
    labels = []
    folder = 'data/training'
    for i, filename in enumerate(os.listdir(folder)):
        if i == 30:
            break
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder,filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # display the image and use roipoly for labeling
            fig, ax = plt.subplots()
            ax.imshow(img)
            my_roi = RoiPoly(fig=fig, ax=ax, color='r')
            
            # get the image mask
            mask = my_roi.get_mask(img)

            for y in range(mask.shape[0]):
                for x in range(mask.shape[1]):
                    if mask[y][x]:
                        pixels.append(img[y][x])
                        labels.append(int(0))
            # display the labeled region and the image mask
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('%d pixels selected\n' % img[mask,:].shape[0])
            
            ax1.imshow(img)
            ax1.add_line(plt.Line2D(my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))
            ax2.imshow(mask)
            
            plt.show(block=True)
    return np.array(pixels), np.array(labels)

blue_pixels, blue_labels = create_blue_data()
np.save("blue_pixels.npy", blue_pixels)
np.save("blue_labels.npy", blue_labels)
other_pixels, other_labels = create_other_data()
np.save("other_pixels.npy", other_pixels)
np.save("other_labels.npy", other_labels)
