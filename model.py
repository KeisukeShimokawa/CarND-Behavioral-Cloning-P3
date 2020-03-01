import os
import csv
import cv2
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import Sequence


class BatchGenerator(Sequence):
    """Custom Batch Generator"""
    
    def __init__(self, lines, batch_size, target_image_dir='/opt/data/', is_both_side=False, is_flip=False):
        self.lines = lines
        self.target_image_dir = target_image_dir
        self.is_both_side = is_both_side
        self.is_flip = is_flip
        if is_both_side: print('left/right camera setup')
        if is_flip: print('flip setup')

        self.batch_size = batch_size
        self.actual_batch_size = batch_size
        if is_both_side: self.actual_batch_size *= 3
        if is_flip: self.actual_batch_size *= 2
        print('actual batch size: {}'.format(self.actual_batch_size))

        self.length = len(lines)
        if is_both_side: self.length *= 3
        if is_flip: self.length *= 2
        print('total data length: {}'.format(self.length))

        self.batches_per_epoch = int((self.length - 1) / self.actual_batch_size) + 1
        print('batches per epoch: {}'.format(self.batches_per_epoch))

    def __getitem__(self, idx):
        batch_from = self.batch_size * idx
        batch_to = batch_from + self.batch_size

        if batch_to > self.length:
            batch_to = self.length

        x_batch, y_batch = self.get_image_and_steering(self.lines[batch_from:batch_to],
                                                      self.target_image_dir,
                                                      self.is_both_side,
                                                      self.is_flip)

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        return x_batch, y_batch

    def __len__(self):
        return self.batches_per_epoch

    def on_epoch_end(self):
        pass

    def get_image_and_steering(self, lines, target_image_dir='/opt/IMG/', is_both_side=True, is_flip=True):
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


def get_lines_from_file(path):
    """Get CSV Data as Python List  

    :param path: path for driving_log.csv  

    :return lines: csv data list
    """
    
    error_txt = "{} does not exist!!".format(path)
    assert os.path.exists(path), error_txt
    
    lines = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines


def combine_multiple_lines(*args):
    """Combine Several Simulator Result 

    :param *args: several csv list 

    :return combine_lines: single csv line combined
    """
    
    print('we got {} lines'.format(len(args)))

    combine_lines = []
    for arg in args:
        combine_lines.extend(arg)

    return combine_lines


def get_all_lines(*paths):
    """Get CSV Data as Python List  

    :param *paths: several path for driving_log.csv

    :return all_lines: all combined list
    """

    lines = []
    for path in paths:
        lines.append(get_lines_from_file(path))

    all_lines = combine_multiple_lines(*lines)
    return all_lines
        
    
def get_fname(path):
    """Get path name  

    :param path: image path  

    :return path: like "/data/IMG/*.jpg"
    """
    return '/'.join(path.split('/')[-3:])


def get_model(im_size, cropping):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=im_size))
    model.add(Cropping2D(cropping=cropping))
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model


def main(conf):
    # get all csv data from several simulation result
    lines = get_all_lines(*conf['data'])
    
    # create train/valid generator
    train_lines, valid_lines = train_test_split(lines, test_size=0.2, 
                                                random_state=0, shuffle=True)
    train_batch_generator = BatchGenerator(train_lines, 
                                           batch_size=conf['batch_size'], 
                                           target_image_dir=conf['target_image_dir'],
                                           is_both_side=conf['is_both_side'],
                                           is_flip=conf['is_flip'])
    valid_batch_generator = BatchGenerator(valid_lines, 
                                           batch_size=conf['batch_size'], 
                                           target_image_dir=conf['target_image_dir'])
    model = get_model(conf['im_size'], 
                      (conf['cropping_height'], conf['cropping_width']))
    # model.compile(loss='mean_absolute_error', optimizer='adam')
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(
               train_batch_generator, 
               steps_per_epoch=train_batch_generator.batches_per_epoch, 
               validation_data=valid_batch_generator, 
               validation_steps=valid_batch_generator.batches_per_epoch,
               epochs=conf['epochs'],
               shuffle=True,
               verbose=1)
    model.save('model.h5')

    
    
if __name__ == '__main__':
    # some Parameters
    with open('config.yaml') as f:
        conf = yaml.load(f.read())
        
    main(conf)
    