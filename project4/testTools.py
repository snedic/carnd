import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
from proj4Tools import calibrate_undistort, display_images

# Read in a calibration images
#images = glob('camera_cal/calibration*.jpg')
img = mpimg.imread('camera_cal/calibration9.jpg')
#plt.imshow(img)
#plt.show(block=True)
display_images([img, img])

# Arrays to store object points and image points from all the images
objPoints = [] # 3D point array in real world space
imgPoints = [] # 2D point array in image space

# Prepare object points, like (0,0,0, (1,0,0), .... (7,5,0)
objP = np.zeros((4*8, 3), np.float32)
objP[:, :2] = np.mgrid[0:8, 0:4].T.reshape(-1, 2) # x, y coordinates

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (8, 4), None)

# If corners are found, add object points, image points
if ret == True:
    print('found corners')
    imgPoints.append(corners)
    objPoints.append(objP)

    # Calibrate and remove the distortion from the image
    undistImg = calibrate_undistort(img, objPoints, imgPoints)

    # draw and display the corners
    imgNew = cv2.drawChessboardCorners(img, (8, 4), corners, ret)

    #plt.show(block=True)
    display_images([img, imgNew, undistImg])
