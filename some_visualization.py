# -*- coding: utf-8 -*-
"""
Created on Tue May 30 01:15:45 2017

@author: Yi Zhu
"""
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog

pix_per_cell = 8
cell_per_block = 2
orient = 9
img_car=mpimg.imread('vehicles/vehicles/GTI_Far/image0003.png')
img_notcar= mpimg.imread('non-vehicles/non-vehicles/Extras/extra40.png')
gray1 = cv2.cvtColor(img_car, cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(img_notcar, cv2.COLOR_RGB2GRAY)
features_car, hog_image_car= hog(gray1, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=True, feature_vector=False)
features_nocar, hog_image_nocar = hog(gray2, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=True, feature_vector=False)
fig = plt.figure()
plt.subplot(121)
plt.imshow(img_car, cmap='gray')
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(hog_image_car, cmap='gray')
plt.title('HOG Visualization')
fig = plt.figure()
plt.subplot(121)
plt.imshow(img_notcar, cmap='gray')
plt.title('Example notCar Image')
plt.subplot(122)
plt.imshow(hog_image_nocar, cmap='gray')
plt.title('HOG Visualization')