import line

from proj4Tools import calibrate_camera
from proj4Tools import hls_saturation_threshold, abs_sobel_thresh, mag_sobel_thresh, dir_sobel_threshold
from proj4Tools import warp_image
from proj4Tools import display_images, display_poly_img, display_binary_image_histo, display_fit_lines
from proj4Tools import sliding_windows_fit_lines, find_line_pixels, get_curve_radius, get_scaled_curve_radius
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import cv2
from glob import glob

# (1) Camera Calibration
mtx, dist = calibrate_camera()


images = glob('test_images/*.jpg')
for fn in images:
    leftLine, rightLine = line.Line(), line.Line()
    img = mpimg.imread(fn)

    # (2) Distortion Correction
    undistImg = cv2.undistort(img, mtx, dist, None, mtx)

    # (3) Color/Gradient Threshold
    s_binary = hls_saturation_threshold(undistImg, thresh=(170, 255))

    # (3) Color/Gradient Threshold
    s_binary = hls_saturation_threshold(undistImg, thresh=(170, 255))

    ksize = 3
    gradx = abs_sobel_thresh(undistImg, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(undistImg, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    magSobel = mag_sobel_thresh(undistImg, sobel_kernel=ksize, thresh=(20, 100))
    dirSobel = dir_sobel_threshold(undistImg, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combo = np.zeros_like(s_binary)
    combo[(s_binary == 1) | (gradx == 1)] = 1

    # (4) Perspective Transform
    srcCoords = np.float32([[578, 460], [705, 460], [1122, 719], [190, 719]]) # [Top Left, Top Right, Bottom Right, Bottom Left]
    #srcCoords = np.float32([[560, 486], [725, 486], [1122, 719], [170, 719]]) # [Top Left, Top Right, Bottom Right, Bottom Left]
    #srcCoords = np.float32([[480, 524], [824, 524], [1122, 719], [190, 719]]) # [Top Left, Top Right, Bottom Right, Bottom Left]
    #srcCoords = np.float32([[540, 480], [725, 480], [1122, 719], [170, 719]]) # [Top Left, Top Right, Bottom Right, Bottom Left]
    dstCoords = np.float32([[180, 0], [1122, 0], [1122, 719], [180, 719]]) # [Top Left, Top Right, Bottom Right, Bottom Left]
    #dstCoords = np.float32([[300, 0], [950, 0], [950, 700], [300, 700]]) # [Top Left, Top Right, Bottom Right, Bottom Left]

    warpImg, Minv = warp_image(combo, srcCoords, dstCoords, inverse=True)

    #fig, ax0 = display_poly_img(img=combo, coords=srcCoords, colorMap='gray')
    #fig, ax1 = display_poly_img(img=warpImg, coords=dstCoords, colorMap='gray', color='blue', fig=fig, subplot=222)
    #plt.show()

    # find the lane lines
    #display_binary_image_histo(warpImg)

    # find the lane lines
    if not (leftLine.detected and rightLine.detected):
        #left_fit, right_fit, out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, leftx, rightx = sliding_windows_fit_lines(warpImg)
        left_fit, right_fit, left_xy, right_xy = sliding_windows_fit_lines(warpImg)
        leftLine.detected = True
        rightLine.detected = True

        #display_fit_lines(warpImg, left_fit, right_fit)
    else:
        left_fit, right_fit, left_xy, right_xy = find_line_pixels(warpImg, left_fit, right_fit)
        #display_fit_lines(warpImg, left_fit, right_fit)

    # Store line values
    leftLine.storeValues(binary_img=warpImg, curr_fit=left_fit, curr_xy=left_xy)
    rightLine.storeValues(binary_img=warpImg, curr_fit=right_fit, curr_xy=right_xy)

    print()
    print(leftLine.radius_of_curvature)
    print(rightLine.radius_of_curvature)
# 4392.7730555
# 10168.0149202

    ploty = np.linspace(0, warpImg.shape[0] - 1, warpImg.shape[0])
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warpImg).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([leftLine.recent_xfitted[-1], ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rightLine.recent_xfitted[-1], ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistImg, 1, newwarp, 0.3, 0)
    plt.imshow(result)
    plt.show()
