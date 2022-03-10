'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import os
import matplotlib.pyplot as plt; plt.ion()
import cv2

class PixelClassifier():
  def __init__(self, data=None):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    self.learning_rate = 0.1
    self.batch_size = 100
    self.tol = 1e-3
    self.max_iter = 50
    if data=="bin":
      self.w = np.array([[1.49678923, -40.96462496, 36.36870051, -4.89983946],
                        [-1.26248478, 41.05639961, -36.45069465, 4.07212496]])
    else:
      self.w = np.array([[31.5568539, -16.22354567, -15.78354851, -0.08625269],
                        [-16.33206885, 31.31506731, -15.69272559, 0.11991397],
                        [-16.218694, -15.92775108, 31.38069252, 0.31214527]])
    pass

  # HELPER FUNCTIONS FOR TRAINING FUNCTION
  def shuffle(self, dataset):
    shuffle_indices = np.random.permutation(len(dataset[0]))
    shuffleX = dataset[0][shuffle_indices]
    shuffleY = dataset[1][shuffle_indices]
    return shuffleX, shuffleY
  
  def one_hot_encoding(self, y, c):
    return np.eye(c)[y-1]

  def softmax(self, z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / e_z.sum(axis=1, keepdims=True)
  
  def generate_minibatches(self, dataset, batch_size):
    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]

  def train(self, dataset=None):
    # CHOOSE BETWEEN DATASETS FOR BIN AND PIXEL
    if dataset=="bin":
      blue_pix = np.load("blue_pixels.npy")
      other_pix = np.load("other_pixels.npy")
      X_train = np.concatenate((blue_pix,other_pix))
      X_train = X_train/255.0
      blue_lab = np.load("blue_labels.npy")
      other_lab = np.load("other_labels.npy")
      y_train = np.concatenate((blue_lab,other_lab))
      classes = 2
    else:
      X_train, y_train = pixel_training_data()
      X_val, y_val = pixel_validation_data()
      classes = 3
    
    # PROCESS DATA BY SHUFFLING AND ADDING BIAS
    X_train, y_train = self.shuffle((X_train,y_train))
    X_train = np.hstack((X_train,np.ones((X_train.shape[0],1))))

    # INITIALIZE WEIGHTS
    self.w = np.random.randn(classes, len(X_train[0]))/3

    # MINIBATCH GRADIENT DESCENT
    train_acc = []
    val_acc = []
    for _ in range(self.max_iter):
      train_acc_batches=[]
      last_w = self.w
      for mini_X, mini_y in self.generate_minibatches((X_train,y_train),batch_size=self.batch_size):
        if not len(mini_X)==self.batch_size:
          continue
        y_hot = self.one_hot_encoding(mini_y,classes).T
        self.w = self.w + self.learning_rate * ((y_hot.T - self.softmax(mini_X @ self.w.T)).T @ mini_X)
        train_acc_batches.append(np.mean(np.argmax(self.softmax(mini_X @ self.w.T), axis=1) == np.argmax(y_hot,axis=1)))
      if np.linalg.norm(last_w - self.w) < self.tol:
        break
      train_acc.append(np.mean(train_acc_batches))
      y_val_pred = self.classify(X_val)
      val_acc.append((y_val_pred==y_val)*1.0/len(y_val))
    plt.plot(train_acc, label='Training')
    plt.plot(val_acc, label='Validation')
    plt.title('Training and Validation Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    ################################################################
    # YOUR CODE AFTER THIS LINE

    # THE COMMENTED LINES ARE AN EXAMPLE OF TRAINING AND EXTRACTING THE WEIGHTS FOR LATER USE
    #self.train(dataset="bin")
    #print(self.w)

    # ADD BIAS AND CLASSIFY
    X = np.hstack((X,np.ones((X.shape[0],1))))
    y = 1 + np.argmax(X @ self.w.T, axis=1).astype(int).reshape(-1)
    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return y

# GENERATE PIXEL TRAINING DATA
def pixel_training_data():
  folder = 'data/training'
  X1 = read_pixels(folder+'/red')
  X2 = read_pixels(folder+'/green')
  X3 = read_pixels(folder+'/blue')
  y1, y2, y3 = np.full(X1.shape[0],1), np.full(X2.shape[0], 2), np.full(X3.shape[0],3)
  X, y = np.concatenate((X1,X2,X3)), np.concatenate((y1,y2,y3))
  return X, y

def pixel_validation_data():
  folder = 'data/validation'
  X1 = read_pixels(folder+'/red')
  X2 = read_pixels(folder+'/green')
  X3 = read_pixels(folder+'/blue')
  y1, y2, y3 = np.full(X1.shape[0],1), np.full(X2.shape[0], 2), np.full(X3.shape[0],3)
  X, y = np.concatenate((X1,X2,X3)), np.concatenate((y1,y2,y3))
  return X, y

# COPY FUNCTION FROM GENERATE RGB DATA TO MAKE TRAINING DATA 
def read_pixels(folder, verbose = False):
  '''
    Reads 3-D pixel value of the top left corner of each image in folder
    and returns an n x 3 matrix X containing the pixel values 
  '''  
  n = len(next(os.walk(folder))[2]) # number of files
  X = np.empty([n, 3])
  i = 0
  
  if verbose:
    fig, ax = plt.subplots()
    h = ax.imshow(np.random.randint(255, size=(28,28,3)).astype('uint8'))
  
  for filename in os.listdir(folder):  
    # read image
    # img = plt.imread(os.path.join(folder,filename), 0)
    img = cv2.imread(os.path.join(folder,filename))
    # convert from BGR (opencv convention) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # store pixel rgb value
    X[i] = img[0,0].astype(np.float64)/255
    i += 1
    
    # display
    if verbose:
      h.set_data(img)
      ax.set_title(filename)
      fig.canvas.flush_events()
      plt.show()

  return X

def accuracy(prediction, labels):
  return np.mean(np.argmax(prediction, axis=1) == np.argmax(labels,axis=1))