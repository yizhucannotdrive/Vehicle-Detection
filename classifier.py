# -*- coding: utf-8 -*-
"""
Created on Mon May 29 00:18:39 2017

@author: Yi Zhu
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from lesson import *
# Define a function to return HOG features and visualization
cars = []
notcars = []
images1 = glob.glob('vehicles/vehicles/GTI_Far/*.png')
for image in images1:
    cars.append(image)
    
images2= glob.glob('vehicles/vehicles/GTI_Left/*.png')
for image in images2:
    cars.append(image)
    
images3= glob.glob('vehicles/vehicles/GTI_MiddleClose/*.png')
for image in images3:
    cars.append(image)

images4= glob.glob('vehicles/vehicles/GTI_Right/*.png')
for image in images4:
    cars.append(image)
    
images5=glob.glob('vehicles/vehicles/KITTI_extracted/*.png')
for image in images5:
    cars.append(image)
    
images6= glob.glob('non-vehicles/non-vehicles/Extras/*.png')
for image in images6:
    notcars.append(image)
    
images7= glob.glob('non-vehicles/non-vehicles/GTI/*.png')
for image in images7:
    notcars.append(image)

### TODO: Tweak these parameters and see how the results change.
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size=(32, 32)
hist_bins=32
t=time.time()
car_features = extract_features(cars, color_space=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
notcar_features = extract_features(notcars, color_space=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
#svc.fit(scaled_X,y)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
import pickle
svc_pickle = { "svc": svc, "scaler": X_scaler , "orient": orient, "pixpercell": pix_per_cell, "cellperblock": cell_per_block, "spatialsize": spatial_size, "histbins": hist_bins}
pickle.dump( svc_pickle, open( "svc_pickle.p", "wb") )