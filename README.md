# CarND-Behavioral Cloning
My behavioral cloning project from Udacity Self Driving Car Nanodegree. This is Project 3. 

Overview
---
In this project, I used deep neural networks on keras (tensorflow as backend) and convolutional neural networks to clone driving behavior through in-house car simulator. The model will output a steering angle to an autonomous vehicle.

Beta version simulator has been used where we can steer a car around a track for data collection. Image data and steering angles are collected to train a neural network and then use this model to drive the car autonomously around the track.

The repository contains: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* autonomous driving video (recorded when in autonomous mode using deep learning model)

[//]: # (Image References)

[image1]: ./examples/model_summary.png "Model Visualization"
[image2]: ./examples/center.png "Example Center image"
[image3]: ./examples/recover.png "Recover from off side"
[image4]: ./examples/beforeflip.png
[image5]: ./examples/afterflip.png

### Dependencies
This project need the conda environment:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

* The simulator can be downloaded from the Udacity classroom. 

### Model Architecture

1. End-to-End Deep Learning architecture from NVIDIA has been used for this employed. More specifically, the model consists of a convolution neural network with 5X5 filter sizes and depth with 24, 36, 48, 3X3 filter sizes and depth with 64. The model includes RELU activation function to introduce nonlinearity, and the data is normalized in the model input using Keras BatchNormalization. 

2. The model contains dropout layers in order to reduce overfitting. I used dropout = 0.4. The model was trained and validated on different dataset with shuffling every epoch to ensure that the model was not overfitting. 

3. The model used an adam optimizer with learning rate specified as 0.002. 

4. The training data was collected through beta simulator which is provided in the course. The data I collected included center lane driving as a main driving hehavior, as well as recovering from the left and  the right to consider recovery behavior. Also I used image flipping to introduce more balanced driving habit. 

5. The model architecture could be seen below, 
![alt text][image1]

### Training Strategy

1. Firstly I collected four to five normal driving laps using beta simulator, to collect both center camera images as well as left and right camera images. Here is an example of image of center lane driving. 

![alt text][image2]

2. I then recorded te vechile recovering from the left and right sides of the road back to center so that the vehicle could learn to how to recover from off side of the road. It is important as otherwise the car will have no clue what to do once getting close or off the road. 

![alt text][image3]

3. I also used flipping images when having non zero steering angle and adjust 0.08 degree on those left and right images, which also help the car to learn to adjust itself to the center of the road. The example of flipping images are below:

![alt text][image4]
![alt text][image5]

4. I separate the datasets into training and validation set with 0.8 vs 0.2. I did not leave for testing set because we will test the vehicle on the simulator automonous mode. 

5. Then I shuffle the dataset every epoch to ensure the model does not take advantage of the time series behavior. Also I applied dropout rate 0.4 to avoid the overfitting. 

6. Generator has been utilized to help memory contraints and gen_batches function was defined for the generator. 



