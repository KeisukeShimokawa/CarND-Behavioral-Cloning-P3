import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Conv2D, AveragePooling2D


def get_lines_from_file(path):
    
    assert os.path.exists(path), "path does not exist!!"
    
    lines = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines
        
    
def get_fname(path):
    return path.split('/')[-1]
    
    
def get_image_and_steering(lines, target_image_dir='/opt/IMG/', is_both_side=True, is_flip=True):
    n_side = 3 if is_both_side else 1
    images = []
    measurements = []
    
    for line in lines:
        correction = 0.2         
        steering = float(line[3])
        
        for i in range(n_side):    
            if i == 1:
                # if camera is set on left
                steering += correction
            elif i == 2:
                # if camera is set on right
                steering -= correction
                
            image_path = target_image_dir + get_fname(line[i])
            assert os.path.exists(image_path), "image path is wrong: {}".format(image_path)
            image = cv2.imread(image_path)
            
            images.append(image)
            measurements.append(steering)
            
            if is_flip:
                images.append(cv2.flip(image, 1))
                measurements.append(steering * -1.0)
            
    return images, measurements


def get_model(im_size, cropping):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=im_size))
    model.add(Cropping2D(cropping=cropping))
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(AveragePooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(AveragePooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(AveragePooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model


if __name__ == '__main__':
    fname = '/opt/driving_log.csv'
    target_image_dir = '/opt/IMG/'
    im_size = (160, 320, 3)
    cropping = ((50, 20), (0, 0))
    is_both_side = False
    is_flip = True
    
    lines = get_lines_from_file(fname)
    print('simulator images: {}'.format(len(lines)))
    
    images, measurements = get_image_and_steering(lines, target_image_dir, is_both_side, is_flip)
    X_train, y_train = np.array(images), np.array(measurements)
    print('training data shape:  {}'.format(X_train.shape))
    print('training label shape: {}'.format(y_train.shape))
    
    model = get_model(im_size, cropping)
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
    model.save('model.h5')