#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 19:46:34 2017

@author: PikeZHAO
"""
# %%
import tensorflow as tf
import csv
import cv2
import numpy as np


batch_size = 100
dropout_factor = 0.3
w_reg = 0.00

lines = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    lines=list(reader)
        
num_frames = len(lines)
print("There are {} frames the input data based on driving_log.csv. ".format(num_frames))
# %% load images (center,left,right) and the steering angle
images=[]
measurements=[]
for line in lines: # num_frame
    for i in range(3): # Three images as Center, Left, Right
        source_path = line[i]
        token = source_path.split("/")
        filename = token[-1]
        local_path = "./IMG/" + filename
        image = cv2.imread(local_path)
        images.append(image)
        if float(line[3]) != 0.0 or i != 0:
            flipped_image=cv2.flip(image,1)
            images.append(flipped_image)
    correction = 0.08
    measurement = float(line[3])
    measurements.append(measurement)
    if measurement != 0.0:
        measurements.append(measurement*-1.0)    
    measurements.append(measurement+correction)
    measurements.append((measurement+correction)*-1.0)
    measurements.append(measurement-correction)
    measurements.append((measurement-correction)*-1.0)
print("Center/Left/Right images are loaded and augmented and flipped!. They are {} frames".format(len(measurements)))
# %% Dealling with the data to be train and validation set
augmented_images=images
augmented_measurements = measurements
from sklearn.utils import shuffle
augmented_images, augmented_measurements = shuffle(augmented_images, augmented_measurements, random_state = 0)
split_index = int(len(augmented_measurements)*0.8)
# split_index2 = int(len(augmented_measurements)*0.25/2)+split_index
X_train, y_train = augmented_images[:split_index],augmented_measurements[:split_index]
X_validation,y_validation = augmented_images[split_index:],augmented_measurements[split_index:]
# X_test,y_test=augmented_images[split_index2:],augmented_measurements[split_index2:]

print("The training set size is ", len(X_train))
print("The validation set size is ", len(X_validation))
# print("The testing set size is ", len(X_test))
# %% 
def gen_batches(X,Y,batch_size):
    index = 0
    while 1:
        X_batch=np.ndarray(shape=(batch_size,160,320,3),dtype=float)
        Y_batch=np.ndarray(shape=(batch_size),dtype=float)        
        for i in range(batch_size):
            if index >= len(X):
                index = 0
                X, Y = shuffle(X, Y, random_state = 0)
            X_batch[i] = X[index]
            Y_batch[i] = Y[index]
            index += 1
        yield (X_batch,Y_batch)

# %% Train the model using NVIDIA End-to-End Learning
w_reg = 0.00
dropout_factor = 0.4
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
#---Reference End to End Learning from NVIDIA
model = Sequential()
# model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))
model.add(BatchNormalization(input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(nb_filter=24,nb_row=5,nb_col=5,subsample=(2,2),border_mode = 'valid',W_regularizer=l2(w_reg)))
model.add(Activation('relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),border_mode = 'valid',W_regularizer=l2(w_reg)))
model.add(Dropout(dropout_factor))
model.add(Activation('relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),border_mode = 'valid',W_regularizer=l2(w_reg)))
model.add(Dropout(dropout_factor))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,subsample=(1,1),border_mode = 'valid',W_regularizer=l2(w_reg)))
model.add(Dropout(dropout_factor))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,subsample=(1,1),border_mode = 'valid',W_regularizer=l2(w_reg)))
model.add(Dropout(dropout_factor))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100,input_shape=(2496,),W_regularizer = l2(w_reg)))
model.add(Activation('relu'))
model.add(Dense(50,W_regularizer = l2(w_reg)))
model.add(Activation('relu'))
model.add(Dense(1,W_regularizer = l2(w_reg)))


model.summary()
# %% Train the model
learning_rate = 0.002
model.compile(optimizer=Adam(lr=learning_rate),loss='mean_squared_error')
# from utils import gen_batches
import math
history = model.fit_generator(gen_batches(X_train,y_train,batch_size),
                    samples_per_epoch=math.ceil(len(X_train)/batch_size)*batch_size,
                    validation_data = gen_batches(X_validation,y_validation,batch_size),
                    nb_val_samples=math.ceil(len(X_validation)/batch_size)*batch_size,
                    nb_epoch=5,
                    verbose = 1,
                    max_q_size=10)

# %% 
model.save('./model_r2.h5')




