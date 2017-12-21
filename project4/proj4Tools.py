import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import ceil

def calibrate_undistort(img, objPts, imgPts):

    # Calibrate the image
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPts, imgPts, img.shape[1::-1], None, None)

    # Undistort the image
    return cv2.undistort(img, mtx, dist, None, mtx)

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
    else:
        M = cv2.getPerspectiveTransform(dst, src)

    # Create warped image - uses linear interpolation
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

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

def hls_saturation_threshold(img, thresh=(0, 255), hlsColorCode=cv2.COLOR_RGB2HLS):
    '''
    :param img:
    :param thresh:
    :param hlsColorCode: image read with mpimg      ---> cv2.COLOR_RGB2HLS
                         image read with cv2.imread ---> cv2.COLOR_BGR2HLS
    :return:
    '''
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, hlsColorCode)

    # 2) Apply a threshold to the S channel
    s = hls[:, :, 2]
    binary_output = np.zeros_like(s)
    binary_output[(s > thresh[0]) & (s <= thresh[1])] = 1

    # 3) Return a binary image of threshold result
    return binary_output

