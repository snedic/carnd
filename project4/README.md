## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images/undistortedChessboard.png "Undistorted"
[image2]: ./images/straight_lines1.jpg "Road Transformed"
[image3]: ./images/undistortedStraight_lines1.jpg "Binary Example"
[image4]: ./images/thresholdstraight_lines1.png "Threshold Example"
[image5]: ./images/warpedstraight_lines1.png "Warp Example"
[image6]: ./images/histowarpedstraight_lines1.png "Fit Visual"
[image7]: ./images/detectedstraight_lines1.png "Output"
[video1]: ./result.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 9 through 51 of "./proj4Tools.py".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objP` is just a replicated array of coordinates, and `objPoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgPoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objPoints` and `imgPoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

File `advancedLaneFinder.py` runs the tests against the images in function run_tests and the video in function run.  The main portion of my code is located in `proj4Tools.py`.

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

to get a result like this one:

![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 518 through 524 in `proj4Tools.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`, which appears in lines 138 through 157 in the file `proj4Tools.py`.  The `warp_image()` function takes as inputs an image (`img`), as well as source (`srcCoords`) and destination (`dstCoords`) points, and whether the image warping needs to be inverted.  This last parameter was found necessary when I was testing the warping with a street sign.  The sign was angled a certain way that required the warping in the opposite direction in order to properly display the text in a flattened manner. I chose to hardcode the source and destination points on line 534 and 535 of `proj4Tools.py` in the following manner:

```python
srcCoords = np.float32( [[578, 460], [705, 460], [1122, 719], [190, 719]])
dstCoords = np.float32( [[180, 0], [1122, 0], [1122, 719], [180, 719]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 578, 460      | 180, 0        | 
| 705, 460      | 1122, 0      |
| 1122, 719     | 1122, 719      |
| 190, 719      | 180, 719        |

I verified that my perspective transform was working as expected by drawing the `srcCoords` and `dstCoords` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I detected the lines pixels using sliding windows and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image6]

Using the binary warped image I used a histogram to identify which pixels were associated with a line on the left half and the right halves of the image.

Note: Once the sliding windows was successful, I was able to find the line pixels more easily for future images by looking within a boundary the previously found lines.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in function `get_curve_radius()` at lines 452 through 258 in my code in `proj4Tools.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 451 where I call function `get_detected_lane()` defined at lines 486 through 505 in my code in `proj4Tools.py` in the function `run_pipeline()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I ran into some issues when the video drove through shaddows.  To avoid misinterpreting the lane, I measure the distance between the two lines.  I had checked several points in the video/test files and found the distance to be in the range of ~890 to 925.  Because of this I added a sanity check for the distance between the lines.  The current threshold is set to a range of 850 to 950.  For images outside these bounds, the values are not stored in the line objects.  I store the last 10 image line details.  If an image is skipped, I still pop values out of the arrays as if I stored the details.  When the arrays size drops below 3, I reassess the line detection with the sliding window technique again.

To improve my pipeline, I could adjust the line distance thresholds to be more tightly bound.  I could also adjust the minimum size of the array to be more than 3 (i.e. 5 or 6) which would force the pipeline to search for pixels more often than it currently does under a bad situation.  On top of this, I could increase the number of previous detections stored.  How many to store would require some additional analysis.
