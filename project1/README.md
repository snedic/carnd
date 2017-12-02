# **Finding Lane Lines on the Road** 

## Writeup Template

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. First, I converted the images to grayscale, then I applied a Gaussian Blur to the images, next I applied Canny.  After that I apply a PolyMask, then the Hough Transform.  Finally I combine the outputs with the original image in order to get the overlay.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by creating sub functions for identification of the left and right lines.  In these I find the mean, med, and std of the slopes associated with each side.  If they points are within a std of the collection of slopes then it is considered when finding the min/max points of the line.  I then draw the lines out from the base of the image to the top point.
    


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when there are a lot of outlier lines, the std value could become too great, thus including lines that are really just cracks or shadows.

Another shortcoming could be the field of view considered in the Hough transform.  The lines are drawn as straight lines.  Given a sharp curve such as a turn at an intersection, the lines would be likely drawn straight through the intersection.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to draw many connected shorter lines to accomodate for turns

Another potential improvement could be to adjust the std limitation to be stricter with regard to what slopes are considered when drawing the left and right lines. 
