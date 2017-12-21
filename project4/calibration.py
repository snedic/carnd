import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
from proj4Tools import calibrate_undistort, display_images

# Read in a calibration images
images = glob('camera_cal/calibration*.jpg')
#img = mpimg.imread('camera_cal/calibration1.jpg')
#plt.imshow(img)
#plt.show(block=True)

# Arrays to store object points and image points from all the images
objPoints = [] # 3D point array in real world space
imgPoints = [] # 2D point array in image space

# Prepare object points, like (0,0,0, (1,0,0), .... (7,5,0)
objP = np.zeros((5*9, 3), np.float32)
objP[:, :2] = np.mgrid[0:9, 0:5].T.reshape(-1, 2) # x, y coordinates

for fname in images:
    # read in each image
    img = mpimg.imread(fname)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 5), None)

    # If corners are found, add object points, image points
    print(fname)
    if ret == True:
        imgPoints.append(corners)
        objPoints.append(objP)

        # Calibrate and remove the distortion from the image
        undistImg = calibrate_undistort(img, objPoints, imgPoints)

        # draw and display the corners
        imgNew = cv2.drawChessboardCorners(img, (8, 6), corners, ret)
 #       plt.imshow(img)

#plt.show(block=True)
display_images([img, imgNew, undistImg])
