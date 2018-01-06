import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from matplotlib.patches import Polygon
from glob import glob
from line import Line
from math import ceil

def calibrate(img, objPts, imgPts):

    # Calibrate the image
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPts, imgPts, img.shape[1::-1], None, None)

    return (mtx, dist)

def calibrate_undistort(img, objPts, imgPts):

    # Calibrate the image
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPts, imgPts, img.shape[1::-1], None, None)

    # Undistort the image
    return cv2.undistort(img, mtx, dist, None, mtx)

def calibrate_camera(calImg='camera_cal/calibration1.jpg'):
    images = glob('camera_cal/calibration*.jpg')

    # Arrays to store object points and image points from all the images
    objPoints = [] # 3D point array in real world space
    imgPoints = [] # 2D point array in image space

    # Prepare object points, like (0,0,0, (1,0,0), .... (7,5,0)
    objP = np.zeros((6*9, 3), np.float32)
    objP[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) # x, y coordinates

    # Read in a calibration images
    for distImg in images:
        img = mpimg.imread(distImg)

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If corners are found, add object points, image points
        if ret is True:
            imgPoints.append(corners)
            objPoints.append(objP)
            #print(distImg)

    # Calibrate and remove the distortion from the image
    img = mpimg.imread(calImg)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, img.shape[1::-1], None, None)

    return (mtx, dist)

def display_images(imgArr, imgLabels=None, isImgGray=None, rows=1):
    if isImgGray is None:
        isImgGray = np.zeros_like(imgArr)

    if type(imgArr) == type(list()) and len(imgArr) > 0:
        n = len(imgArr)
        f, ax = plt.subplots(rows, ceil(n/rows), figsize=(24, 9))
        f.tight_layout()
        for i in range(n):
            colorMap = 'gray' if isImgGray[i] else None

            if rows > 1:
                imgsPerRow = ceil(n/rows)
                axis = ax[int(i / imgsPerRow)][i % imgsPerRow]
            else:
                axis = ax[i]

            axis.imshow(imgArr[i], cmap=colorMap)
            if imgLabels is None:
                axis.set_title('Image {}'.format(i))
            else:
                axis.set_title(imgLabels[i])
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        plt.show()
    else:
        raise Exception

def display_poly_img(img, coords, isFilled=False, color='red', fig=None, subplot=111, colorMap=None):
    poly = Polygon(xy=coords, closed=True, fill=isFilled, edgecolor=color)
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(subplot, aspect='equal')
    ax.imshow(img, cmap=colorMap)
    ax.add_patch(poly)

    return fig, ax
    #plt.show()

def display_binary_image_histo(img):
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    plt.plot(histogram)
    plt.imshow(img, cmap='gray')
    plt.show()

def display_fit_lines(warpedImage, left_fit, right_fit):#, nonzerox, nonzeroy, right_lane_inds, left_lane_inds, outImage=None ):
    #if outImage is None:
    #    outImage = warpedImage

    # Generate x and y values for plotting
    ploty = np.linspace(0, warpedImage.shape[0] - 1, warpedImage.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    #outImage[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #outImage[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(warpedImage, cmap='gray')
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()

def display_lane_detection(img, undistImg, warpedImage, Minv, leftLine, rightLine):
    ploty = np.linspace(0, warpedImage.shape[0] - 1, warpedImage.shape[0])
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warpedImage).astype(np.uint8)
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


def warp_image(img, srcCrds, dstCrds, inverse=False):
    # Define calibration box in source (original) and destination (desired or warped) coordinates
    img_size = (img.shape[1], img.shape[0])

    # Source coordinates
    src = np.float32(srcCrds)

    # Desired coordinates
    dst = np.float32(dstCrds)

    # Compute the perspective transform, M
    if inverse:
        M = cv2.getPerspectiveTransform(src, dst) # computes the inverse
        Minv = cv2.getPerspectiveTransform(dst, src)
    else:
        M = cv2.getPerspectiveTransform(dst, src)
        Minv = cv2.getPerspectiveTransform(src, dst) # computes the inverse

    # Create warped image - uses linear interpolation
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR), Minv

def corners_unwarp(img, nx, ny, mtx, dist, dstTLCoords=[100, 100]):
    # Pass in your image into this function

    # 1) Undistort using mtx and dist
    uImg = cv2.undistort(img, mtx, dist, None, mtx)

    # 2) Convert to grayscale
    gray = cv2.cvtColor(uImg, cv2.COLOR_BGR2GRAY)

    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    # 4) If corners found:
    if ret:
        imgSize = (gray.shape[1], gray.shape[0])

        # a) draw corners
        cv2.drawChessboardCorners(uImg, (8, 6), corners, ret)

        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
        # Note: you could pick any four of the detected corners
        # as long as those four corners define a rectangle
        # One especially smart way to do this would be to use four well-chosen
        # corners that were automatically detected during the undistortion steps
        # We recommend using the automatic detection of corners in your code
        srcCoords = [corners[0][0], corners[nx - 1][0], corners[-1][0], corners[-nx][0]]
        src = np.float32(srcCoords)

        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        dstTLCoords = [100, 100]

        dstCoords = [dstTLCoords,
                     [imgSize[0] - dstTLCoords[0], dstTLCoords[1]],
                     [imgSize[0] - dstTLCoords[0], imgSize[1] - dstTLCoords[1]],
                     [dstTLCoords[0], imgSize[1] - dstTLCoords[1]]]
        dst = np.float32(dstCoords)

        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)

        # e) use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(uImg, M, imgSize, flags=cv2.INTER_LINEAR)

    return warped, M

def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(0, 255), grayColorCode=cv2.COLOR_RGB2GRAY):
    ''' 
    :param img:
    :param orient: 
    :param thresh_min: 
    :param thresh_max: 
    :param grayColorCode: image read with mpimg      ---> cv2.COLOR_RGB2Gray
                          image read with cv2.imread ---> cv2.COLOR_BGR2GRAY
    :return: 
    '''

    # Grayscale
    gray = cv2.cvtColor(img, grayColorCode)

    # Apply cv2.Sobel()
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else: # orient == 'x'
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)

    # Take the absolute value of the output from cv2.Sobel()
    absSobel = np.absolute(sobel)

    # Scale the result to an 8-bit range (0-255)
    scaledSobel = np.uint8(255 * absSobel / np.max(absSobel))

    # Apply lower and upper thresholds
    sxbinary = np.zeros_like(scaledSobel)

    sxbinary[(scaledSobel >= thresh[0]) & (scaledSobel <= thresh[1])] = 1

    return sxbinary

def mag_sobel_thresh(img, sobel_kernel=3, thresh=(0, 255), grayColorCode=cv2.COLOR_RGB2GRAY):
    '''
    :param img:
    :param sobel_kernel:
    :param mag_thresh:
    :param grayColorCode: image read with mpimg      ---> cv2.COLOR_RGB2Gray
                          image read with cv2.imread ---> cv2.COLOR_BGR2GRAY
    :return:
    '''

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, grayColorCode)

    # 2) Take the gradient in x and y separately
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude
    magSobel = np.sqrt(sobelX ** 2 + sobelY ** 2)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaledSobel = np.uint8(255 * magSobel / np.max(magSobel))

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaledSobel)

    # 6) Return this mask as your binary_output image
    binary_output[(scaledSobel >= thresh[0]) & (scaledSobel <= thresh[1])] = 1

    return binary_output


def dir_sobel_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2), grayColorCode=cv2.COLOR_RGB2GRAY):
    '''
    :param img:
    :param sobel_kernel:
    :param thresh:
    :param grayColorCode: image read with mpimg      ---> cv2.COLOR_RGB2Gray
                          image read with cv2.imread ---> cv2.COLOR_BGR2GRAY
    :return:
    '''

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, grayColorCode)

    # 2) Take the gradient in x and y separately
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    absSobelX = np.absolute(sobelX)
    absSobelY = np.absolute(sobelY)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dirSobel = np.arctan2(absSobelY, absSobelX)

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(dirSobel)

    # 6) Return this mask as your binary_output image
    binary_output[(dirSobel >= thresh[0]) & (dirSobel <= thresh[1])] = 1

    return binary_output

def hls_saturation_threshold(img, thresh=(170, 255), hlsColorCode=cv2.COLOR_RGB2HLS):
    '''
    :param img:
    :param thresh: values between 0 and 1
    :param hlsColorCode: image read with mpimg      ---> cv2.COLOR_RGB2HLS
                         image read with cv2.imread ---> cv2.COLOR_BGR2HLS
    :return:
    '''
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, hlsColorCode).astype(np.float)

    # 2) Apply a threshold to the S channel
    s = hls[:, :, 2]
    binary_output = np.zeros_like(s)
    binary_output[(s >= thresh[0]) & (s <= thresh[1])] = 1

    # 3) Return a binary image of threshold result
    return binary_output

def sliding_windows_fit_lines(binary_warped):

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])


    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin


        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)


        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]


        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    #return left_fit, right_fit, out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, leftx, rightx
    return left_fit, right_fit, (leftx, lefty), (rightx, righty)


def find_line_pixels(warpedImage, left_fit, right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = warpedImage.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    #ploty = np.linspace(0, warpedImage.shape[0] - 1, warpedImage.shape[0])
    #left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    #right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    #return left_fit, right_fit, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, leftx, rightx
    return left_fit, right_fit, (leftx, lefty), (rightx, righty)

def get_curve_radius(warpedImage, left_fit, right_fit):
    ploty = np.linspace(0, warpedImage.shape[0] - 1, warpedImage.shape[0])
    y_eval = np.max(ploty)

    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    return left_curverad, right_curverad


def get_scaled_curve_radius(warpedImage, left_fit, right_fit):
    ploty = np.linspace(0, warpedImage.shape[0] - 1, warpedImage.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30. / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return left_curverad, right_curverad
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m

def get_detected_lane(img, undistImg, warpedImage, Minv, leftLine, rightLine):
    ploty = np.linspace(0, warpedImage.shape[0] - 1, warpedImage.shape[0])
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warpedImage).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    #pts_left = np.array([np.transpose(np.vstack([leftLine.recent_xfitted[-1], ploty]))])
    #pts_right = np.array([np.flipud(np.transpose(np.vstack([rightLine.recent_xfitted[-1], ploty])))])
    pts_left = np.array([np.transpose(np.vstack([leftLine.best_fit, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rightLine.best_fit, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undistImg, 1, newwarp, 0.3, 0)

    # Add text to the image
    font = cv2.FONT_ITALIC
    txtLines = [''
        '{:.3f} m --- Distance from center --- {:.3f} m'.format(leftLine.line_base_pos, rightLine.line_base_pos),
        '{:.1f} m --- Radius of Curvature --- {:.1f} m'.format(leftLine.radius_of_curvature, rightLine.radius_of_curvature)
                ]
    for i in range(len(txtLines)):
        txt = txtLines[i]
        cv2.putText(result, text=txt.strip(), org=(int((result.shape[1] - 650)/2), result.shape[0]-10-40*i),
                    fontFace=font, fontScale=0.8, color=(255, 255, 255), thickness=2)

    return result



def run_pipeline(img, mtx, dist, leftLine, rightLine):

    # (2) Distortion Correction
    undistImg = cv2.undistort(img, mtx, dist, None, mtx)

    # (3) Color/Gradient Threshold
    s_binary = hls_saturation_threshold(undistImg, thresh=(170, 255))

    # (3) Color/Gradient Threshold
    s_binary = hls_saturation_threshold(undistImg, thresh=(170, 255))

    ksize = 3
    gradx = abs_sobel_thresh(undistImg, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    #grady = abs_sobel_thresh(undistImg, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    #magSobel = mag_sobel_thresh(undistImg, sobel_kernel=ksize, thresh=(20, 100))
    #dirSobel = dir_sobel_threshold(undistImg, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combo = np.zeros_like(s_binary)
    combo[(s_binary == 1) | (gradx == 1)] = 1

    # (4) Perspective Transform [Top Left, Top Right, Bottom Right, Bottom Left]
    srcCoords = np.float32( [[578, 460], [705, 460], [1122, 719], [190, 719]])
    dstCoords = np.float32( [[180, 0], [1122, 0], [1122, 719], [180, 719]])

    warpImg, Minv = warp_image(combo, srcCoords, dstCoords, inverse=True)

    # find the lane lines
    if not (leftLine.detected and rightLine.detected):
        left_fit, right_fit, left_xy, right_xy = sliding_windows_fit_lines(warpImg)
    else:
        left_fit, right_fit, left_xy, right_xy = find_line_pixels(warpImg, leftLine.current_fit, rightLine.current_fit)

    # Sanity Check
    lane_width = np.mean(right_xy[0]) - np.mean(left_xy[0])
    if lane_width > 880. and lane_width < 930.:
        # Store line values
        leftLine.store_values(binary_img=warpImg, curr_fit=left_fit, curr_xy=left_xy)
        rightLine.store_values(binary_img=warpImg, curr_fit=right_fit, curr_xy=right_xy)
    else:
        leftLine.frame_skipped()
        rightLine.frame_skipped()

    return get_detected_lane(img=img, undistImg=undistImg, warpedImage=warpImg, Minv=Minv, leftLine=leftLine, rightLine=rightLine)


