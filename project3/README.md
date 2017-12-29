# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* helpers.py containing helper functions used by my models
* generators.py containing the generator used by my models
* drive.py for driving the car in autonomous mode
* nvidia.h5 containing a trained convolution neural network 
* readme.md or writeup_report.pdf summarizing the results
* run1.mp4 containing a video of a successful run around the track
* msePerEpoch_nvidia.png containing a graph of the training/validation error values
* training.log containing the output of my final training

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py nvidia.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is an implemenation of the suggested nvidia model.  https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

My model consists of a 3 convolutional layers with a 5x5 kernel, 2 convolutional layeres with a 3x3 kernel, followed by 3 fully connected layers and a final output layer.  Eachlayer also includes dropout with a rate of 0.25.  See model.py lines 52-71.

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers at every layer throughout the model in order to reduce overfitting (model.py lines 52-71). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 16-29). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. (see run1.mp4)

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 74).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving the course both directions.  I also used all three camera angles with a correction of 0.15.  I calculated the value to be 0.148 by estimating the angle from each side camera to be ~26.6.  I opted to round the value to 0.15.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had larger images than were provided in the training set. I then preprocessed this data by resizing the images to 323x32x3, converting to grayscale, and applying the same normalization as applied during training and validation steps.

Having pulled the images from various websites, I noticed a few potential issues.  First, the images were of various dimensions, so resizing them all to 32x32x3 most likely had different levels of data loss.  Second, the images had different angles on the signs, with varying backgrounds (clouds, trees, etc.) that could impact the overall ability of the model to classify the images.  I did not crop the images at all.  Doing this could help eliminate background oise in the images thus helping the model to more easily identifying the proper label.  Finally, some of the images used were from websites that included watermarks.  These watermarks are embedded all over the images.  Resizing the images should have reduced the negative impact of the watermarks, however, they would still degrade the images and make it harder for a model to classify them.


Prior to training, I randomly shuffled the data set and used the validation dataset provided. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 100 and 200 for a couple models and between 300 and 400 for another as is evidenced by the various charts in my notebook. I used an adam optimizer so that manually training the learning rate wasn't necessary.
