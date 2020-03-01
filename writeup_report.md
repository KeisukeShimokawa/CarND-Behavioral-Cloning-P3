# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 16 and 64 (model.py lines 190-203) 

|      Layer      |                 Description                  |
|:---------------:|:--------------------------------------------:|
|      Input      |             160x320x3 RGB image              |
|     Lambda      |             normalize RGB image              |
|    Cropping     |       (height=(70, 20), width=(0, 0))        |
| Convolution 3x3 | 16 filters 1x1 stride, same padding, outputs |
|      RELU       |                                              |
|   Max pooling   |             2x2 stride,  outputs             |
| Convolution 3x3 | 32 filters 1x1 stride, same padding, outputs |
|      RELU       |                                              |
|   Max pooling   |             2x2 stride,  outputs             |
| Convolution 3x3 | 64 filters 1x1 stride, same padding, outputs |
|      RELU       |                                              |
|   Max pooling   |             2x2 stride,  outputs             |
|     Linear      |                     128                      |
|     Dropout     |                     0.5                      |
|      RELU       |                                              |
|     Linear      |                      64                      |
|     Dropout     |                     0.5                      |
|      RELU       |                                              |
|     Linear      |                      1                       |



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

We use 5 data created at different times to avoid overfitting. In order to evaluate the performance of the model, course data for one round is created.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 227).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

| dataset          | description         |
|------------------|---------------------|
| data             | 3 laps              |
| data-normal1     | more 3laps          |
| data-rev         | reverse 3 laps      |
| data-normal-rev1 | more reverse 3 laps |
| data-left        | winding the course  |
| data-valid       | validation dataset  |


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Initially I created data on driving a car along the center of the course. We also created data that goes around the same course in reverse along the center.

Using these data, we trained the model presented above and tried this model in autonomous driving mode. The car drove well in places where the course was somewhat straight, but when we approached a curve, we could not turn well and left the course.

The reason that an autonomous car was unable to turn a curve is that the data used when training the model was only moving along the center of the course, and the behavior of the car when it was about to leave the course was It was not included.

In order to solve this problem, we have created new data to recommend a car while winding around the course. In addition, since the ratio of this data in the entire data was about 20%, it was slightly over-fitted to the training data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.

|      Layer      |                 Description                  |
|:---------------:|:--------------------------------------------:|
|      Input      |             160x320x3 RGB image              |
|     Lambda      |             normalize RGB image              |
|    Cropping     |       (height=(70, 20), width=(0, 0))        |
| Convolution 3x3 | 16 filters 1x1 stride, same padding, outputs |
|      RELU       |                                              |
|   Max pooling   |             2x2 stride,  outputs             |
| Convolution 3x3 | 32 filters 1x1 stride, same padding, outputs |
|      RELU       |                                              |
|   Max pooling   |             2x2 stride,  outputs             |
| Convolution 3x3 | 64 filters 1x1 stride, same padding, outputs |
|      RELU       |                                              |
|   Max pooling   |             2x2 stride,  outputs             |
|     Linear      |                     128                      |
|     Dropout     |                     0.5                      |
|      RELU       |                                              |
|     Linear      |                      64                      |
|     Dropout     |                     0.5                      |
|      RELU       |                                              |
|     Linear      |                      1                       |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![check](https://i.gyazo.com/066519d2bf8c03c363ba68c30ada4b6f.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![center](https://i.gyazo.com/283c6b4759bdd121c359a30b3aca174b.jpg)
![left](https://i.gyazo.com/ddbc733104d6c424140fa578e6063cf4.jpg)
![right](https://i.gyazo.com/678282b9ebbcf0308db28ad56e9c8aca.jpg)

Then I repeated this process on track three in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

we have created new data to recommend a car while winding around the course.

![winding1](https://i.gyazo.com/9d49d2f86dbecd82bdc67147c71ac922.jpg)
![winding2](https://i.gyazo.com/be8c803f6f4bd5c5e98c4b0329cb6ec7.jpg)

After the collection process, I had about 25,000 number of data points.


To validate the model, all data was randomly shuffled into training and validation data.

However, with the random shuffling method, you do not know if the model really fits the unknown data. Therefore, a different data set was newly created as verification data.

This is final result!

[autonomous driving](https://www.dropbox.com/s/xv0jjqtqsiolyb6/run1.mp4)
