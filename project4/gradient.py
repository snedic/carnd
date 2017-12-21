from proj4Tools import abs_sobel_thresh, mag_sobel_thresh, dir_sobel_threshold
from proj4Tools import display_images
import matplotlib.image as mpimg
import numpy as np

# Read the image
image = mpimg.imread('images/signs_vehicles_xygrad.png')

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_sobel_thresh(image, sobel_kernel=ksize, thresh=(30, 100))
dir_binary = dir_sobel_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

# combine the results
combo = np.zeros_like(image)
combo[(gradx == 1) & (grady == 1) & (mag_binary == 1) & (dir_binary == 1)] = 1

# display each result
display_images(imgArr=[image, gradx, grady, mag_binary, dir_binary, combo],
               imgLabels=['Original','Sobel X', 'Sobel Y','Sobel Magnitude','Sobel Direction', 'Combination'],
               isImgGray=[0, 1, 1, 1, 1, 1],
               rows=2)

