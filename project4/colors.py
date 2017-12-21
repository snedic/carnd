from proj4Tools import hls_saturation_threshold
from proj4Tools import display_images
import matplotlib.image as mpimg
import numpy as np

# Read the image
image = mpimg.imread('images/signs_vehicles_xygrad.png')

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
s_binary = hls_saturation_threshold(image, thresh=(90, 255))

# combine the results

# display each result
display_images(imgArr=[image, s_binary],
               imgLabels=['Original', 'HLS Saturation Threshold'],
               isImgGray=[0, 1],
               rows=1)