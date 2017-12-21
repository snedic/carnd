from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import cv2
from proj4Tools import display_images, warp_image

# image downloaded from https://legallogik.com/wp-content/uploads/2017/05/TicketAide-Ticket-Stop-Sign.jpg
fn = './images/stopSign.jpg'

# read and display the original image
img = mpimg.imread(fn)
plt.imshow(img)

# image coords - [Top Left, Bottom Left, Top Right, Bottom Right]
srcCoords = [[971, 288], [977, 443], [1377, 142], [1394, 330]]
dstCoords = [[971, 288], [971, 443], [1388, 288], [1388, 443]]

for coord in srcCoords:
    plt.plot(coord[0], coord[1], '.')

plt.show()

wImg = warp_image(img, srcCoords, dstCoords, inverse=True)
display_images([img, wImg])




import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# MODIFY THIS FUNCTION TO GENERATE OUTPUT
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    uImg = cv2.undistort(img, mtx, dist, None, mtx)
    # 2) Convert to grayscale
    gray = cv2.cvtColor(uImg, cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
    # 4) If corners found:
    if ret:
            # a) draw corners
            img = cv2.drawChessboardCorners(img, (8, 6), corners, ret)
            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
                 #Note: you could pick any four of the detected corners
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code
            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
            # e) use cv2.warpPerspective() to warp your image to a top-down view
    #delete the next two lines
    M = None
    warped = np.copy(img)
    return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

