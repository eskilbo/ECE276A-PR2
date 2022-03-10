'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import cv2
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation,binary_erosion
import matplotlib.pyplot as plt

class BinDetector():
	def __init__(self):
		'''
			Initilize your bin detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		# USE WEIGHTS FOUND AFTER TRAINING
		self.w = np.array([[1.49678923, -40.96462496, 36.36870051, -4.89983946],
				[-1.26248478, 41.05639961, -36.45069465, 4.07212496]])

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE

		# CREATE EMPTY IMAGE AND FLATTEN IMAGE WHILE RETAINING RGB
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		X = np.empty([img.shape[0]*img.shape[1],3])
		X = (img.astype(np.float64)/255).reshape(img.shape[0]*img.shape[1],3)
		mask_img = self.classify(X).reshape(img.shape[:2])
		mask_img = (mask_img == 1).astype(int)
		# YOUR CODE BEFORE THIS LINE
		################################################################

		return mask_img

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		boxes = []
		mask_img = binary_erosion(img, iterations=7)
		mask_img = binary_dilation(mask_img, iterations=9)
		labels = label(mask_img)
		region_props = regionprops(labels)
		num_pixels = img.shape[0]*img.shape[1]
		for region in region_props:

			# SIMILARITY CHECK, REMOVES UNLIKELY AREAS FROM CONTENTION
			area = region.area
			if area/num_pixels > 0.5 or area/num_pixels < 0.005:
				continue
			(y1,x1,y2,x2) = region.bbox
			if abs(x2-x1)/abs(y2-y1) < 0.33 or abs(x2-x1)/abs(y2-y1) > 1.5:
				continue
			# ONLY APPEND THE BOXES WHO SATISFY ALL CRITERIA
			boxes.append([x1,y1,x2,y2])
		
		# YOUR CODE BEFORE THIS LINE
		################################################################
		return boxes
	def classify(self,X):
		'''
	    Classify a set of pixels into recycle bin blue or not blue
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with blue or not blue
    	'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		X = np.hstack((X,np.ones((X.shape[0],1))))
		y = 1 + np.argmax(X @ self.w.T, axis=1).astype(int).reshape(-1)
		# YOUR CODE BEFORE THIS LINE
		################################################################
		return y