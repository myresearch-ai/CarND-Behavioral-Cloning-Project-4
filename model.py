# -*- coding: utf-8 -*-
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda#, ELU
#from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
#from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import img_to_array, load_img

#from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle

import pandas as pd
import numpy as np
import cv2
#import math

#import matplotlib.image as mpimg

flags = tf.app.flags
FLAGS = flags.FLAGS

# SET SEEDS
seed = 42
np.random.seed(seed)
tf.random.set_random_seed(seed) 

# DEFINE FLAGS VARIABLES
# RESOURCE: https://stackoverflow.com/questions/33932901/whats-the-purpose-of-tf-app-flags-in-tensorflow
flags.DEFINE_float("steering_adjustment", 0.27, "Adjustment angle")
flags.DEFINE_integer("epochs", 30, "Number of training epochs")
flags.DEFINE_integer("batch_size", 64, "Batch size")

# Set img params
ROWS, COLS, C = (64, 64, 3)
TARGET_SIZE   = (64, 64)

# =============================================================================
# center = [x for x in df["center"]]
# center_recover = center.copy()
# 
# left = [x for x in df["left"]]
# right= [x for x in df["right"]]
# 
# steering = [x for x in df["steering"]]
# steering_recovery = steering.copy()
# =============================================================================
# HELPER FUNCTIONS FOR IMAGE DATA TRANSFORMATIONS & PREPROCESSING
def _image_brightness(img):
    """
    Manipulates brightness/contrast of an image
    :param img: Input image
    :return: output image with transformed or adjusted brightness
    """
    shifted_img = img + 1.0
    img_max_value = max(shifted_img.flatten())
    max_coef = 2.0/img_max_value
    min_coef = max_coef - 0.1
    coef = np.random.uniform(min_coef, max_coef)
    dst = shifted_img * coef
    return dst

def _image_warpaffine(img):
    """
    Applies warp affine transformation to a given image
    :param img: Input image
    :return: outputs a tranformed image with warp affine applied to it.
    """
    rows,cols = img.shape[0:2]

    # random scaling coefficients
    random_x = np.random.rand(3) - 0.5
    random_x *= cols * 0.04 
    random_y = np.random.rand(3) - 0.5
    random_y *= rows * 0.04

    # 3 starting points for transform
    x1 = cols/4
    x2 = 3*cols/4
    y1 = rows/4
    y2 = 3*rows/4
    points_in = np.float32([[y1,x1],[y2,x1],[y1,x2]])
    points_out = np.float32([[y1+random_y[0],x1+random_x[0]],[y2+random_y[1],x1+random_x[1]],
                             [y1+random_y[2],x2+random_x[2]]])
    M = cv2.getAffineTransform(points_in,points_out)
    dst = cv2.warpAffine(img,M,(cols,rows))
    dst = dst[:,:,np.newaxis]

    return dst

def _image_perspective(img):
    """
    Applies perspective tranformation to a given image
    :param img: Input image
    :return: Returns an image with warp perspective applied to it
    """
    # Specify desired outputs size
    width, height = img.shape[0:2]
    # Specify congugate x, y coordinates
    pixels = np.random.randint(-2,2)
    points_in = np.float32([[pixels,pixels],[width-pixels,pixels],[pixels,height-pixels],[width-pixels,height-pixels]])
    points_out= np.float32(([[0,0],[width,0],[0,height],[width,height]]))
    # Perform perspective transformation using cv2
    M = cv2.getPerspectiveTransform(points_in,points_out)
    dst = cv2.warpPerspective(img,M,(width,height))
    dst = dst[:,:,np.newaxis]
    
    return dst

def _image_resize(img):
    """
    Resizes an image
    :param img: Input image
    :return: Returns an image of size equal to preset image size
    """
    return cv2.resize(img, TARGET_SIZE)

def _image_crop_and_resize(img):
    """
    Crops an image
    :param img: Input image of dimension 160x320x3
    :return: A cropped and resized image of dimension 64x64x3 
    """
    cropped = img[55:135,:,:]
    dst = _image_resize(cropped)
    
    return dst

def preprocess_image(img):
    image = _image_crop_and_resize(img)
    image = image.astype(np.float32)

    #Normalize image
    dst = image/255.0 - 0.5
    return dst


def get_augmented_row(row):
    steering = row['steering']

    # randomly choose the camera to take the image from
    camera = np.random.choice(['center', 'left', 'right'])

    # adjust the steering angle for left anf right cameras
    if camera == 'left':
        steering += 0.25
    elif camera == 'right':
        steering -= 0.25

    image = load_img("data/" + row[camera].strip())
    image = img_to_array(image)

    # decide whether to horizontally flip the image:
    # This is done to reduce the bias for turning left that is present in the training data
    flip_prob = np.random.random()
    if flip_prob > 0.5:
        # flip the image and reverse the steering angle
        steering = -1*steering
        image = cv2.flip(image, 1)

    # Apply brightness augmentation
    image = _image_brightness(image)

    # Crop, resize and normalize the image
    dst = preprocess_image(image)
    return dst, steering

def get_data_generator(data_frame, batch_size=32):
    N = data_frame.shape[0]
    batches_per_epoch = N // batch_size

    i = 0
    while(True):
        start = i*batch_size
        end = start+batch_size - 1

        X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0

        # slice a `batch_size` sized chunk from the dataframe
        # and generate augmented data for each row in the chunk on the fly
        for index, row in data_frame.loc[start:end].iterrows():
            X_batch[j], y_batch[j] = get_augmented_row(row)
            j += 1

        i += 1
        if i == batches_per_epoch - 1:
            # reset the index so that we can cycle over the data_frame again
            i = 0
        yield X_batch, y_batch

# PREP DATA
df = pd.read_csv("./data/driving_log.csv", skiprows = [0], names = ["center", "left", "right", "steering", "throttle", "brake", "speed"])
# SHUFFLE DATA
df = df.sample(frac=1).reset_index(drop=True)
# SPLIT: 80-20
training_split = 0.8
num_rows_training = int(df.shape[0]*training_split)
training_data = df.loc[0:num_rows_training-1]
validation_data = df.loc[num_rows_training:]
# SPACE/MEM MGMNT 
del df

# SET FLAG?
#batch_size = 64
training_generator   = get_data_generator(training_data, batch_size = FLAGS.batch_size)
validation_generator = get_data_generator(validation_data, batch_size = FLAGS.batch_size)
samples_per_epoch = (20000//FLAGS.batch_size)*FLAGS.batch_size

def main(_):
# Training Architecture: inspired by NVIDIA architecture #
  input_shape = (64,64,3)
  model = Sequential()
  model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
  model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Convolution2D(64, 3, 3, border_mode='same', subsample = (2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Flatten())
  model.add(Dense(80, W_regularizer = l2(0.001)))
  model.add(Dropout(0.5))
  model.add(Dense(40, W_regularizer = l2(0.001)))
  model.add(Dropout(0.5))
  model.add(Dense(16, W_regularizer = l2(0.001)))
  model.add(Dropout(0.5))
  model.add(Dense(10, W_regularizer = l2(0.001)))
  model.add(Dense(1, W_regularizer = l2(0.001)))
  adam = Adam(lr = 0.0001)
  model.compile(optimizer= adam, loss='mse', metrics=['accuracy'])
  model.summary()
  model.fit_generator(training_generator, validation_data=validation_generator,
                        samples_per_epoch=samples_per_epoch, nb_epoch=2, nb_val_samples=3000)

  print('Done Training')

###Saving Model and Weights###
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
    json_file.write(model_json)
  model.save_weights("model.h5")
  print("Saved model to disk")

if __name__ == '__main__':
  tf.app.run()