# -*- coding: utf-8 -*-
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout, Conv2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
            
class AutonomousDriving():
    def __init__(self):
        """
        Training class for Udacity's Bahavorial Cloning Project.
        """
        self.path = ''
        self.steering_offset = 0.2
        self.model = None
        self.training_samples = []
        self.validation_samples = []

    def build_model(self):
        """
        Inspired by NVIDIA Model architecture.
        """
        self.model = Sequential()
        
        # Normalization
        self.model.add(Lambda(lambda x: (x/127.5) - 1., input_shape=(160,320,3)))
        
        # Cropping2D - (top, bottom), (left, right)
        self.model.add(Cropping2D(cropping=((60,25),(0,0))))
        
        # Layer 1 - Convolution
        self.model.add(Conv2D(24, (5,5), strides=(2,2), activation="relu"))
        
        # Layer 2 - Convolution
        self.model.add(Conv2D(36, (5,5), strides=(2,2), activation="relu"))
        
        # Layer 3 - Convolution
        self.model.add(Conv2D(48, (5,5), strides=(2,2), activation="relu"))
        
        # Layer 4 - Convolution
        self.model.add(Conv2D(64, (3,3), activation="relu"))
        
        # Layer 5 - Convolution 
        self.model.add(Conv2D(64, (3,3), activation="relu"))
        
        self.model.add(Flatten())
        
        # Layer 6 - Fully connected
        self.model.add(Dense(100))
        self.model.add(Activation('relu'))
        
        # Dropout
        self.model.add(Dropout(0.25))
        
        # Layer 7 - Fully connected
        self.model.add(Dense(50))
        self.model.add(Activation('relu'))
        
        # Layer 8 - Fully connected
        self.model.add(Dense(10))
        self.model.add(Activation('relu'))
        
        # Layer 9 - Fully connected
        self.model.add(Dense(1))
        
        # Compile
        self.model.compile(loss='mse', optimizer='adam')
    
    def import_data(self, path, validation_size=0.2):
        """
        Reads in the driving_log.csv file. Each line contains the path to the center, left, and 
        right image as well as the corresponding steering angle.
        |center|left|right|steering|throttle|brake|speed|
        -------|----|-----|--------|--------|-----|-----|
        |aaaaa|bbbbb|ccccc|dddddddd|eeeeeeee|fffff|ggggg|
        
        Also partitions the data into a training and a validation set based on the specified ratio.
        """
        # read in samples from csv
        data = []
        self.path = path
        csv_path = os.path.join(path,'driving_log.csv')
        
        if not os.path.isfile(csv_path):
            raise Exception('Error reading driving log. No such file: {}'.format(csv_path))
        
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for line in reader:
                data.append(line)
            
        # split samples into train and validation set
        self.training_samples, self.validation_samples = train_test_split(data, test_size=validation_size)
        
    def generator(self, data, batch_size):
        """
        Generate batch iterator.
        """
        
        num_samples = len(data)
        
        while True:
            shuffle(data)

            # extract batch from samples
            for offset in range(0, num_samples, batch_size):
                batch_samples = data[offset:offset + batch_size]
                images, steering_angles = [], []

                # process all samples in batch
                for batch_sample in batch_samples:
                    augmented_images, augmented_angles = self.process_batch_sample(batch_sample)
                    images.extend(augmented_images)
                    steering_angles.extend(augmented_angles)

                X_train, y_train = np.array(images), np.array(steering_angles)
                yield shuffle(X_train, y_train)
    
    def process_batch_sample(self, batch_sample):
        """
        Returns the images and corresponding steering angles for each batch.
        """
        steering_angle = np.float32(batch_sample[3])
        images, steering_angles = [], []

        # Read center (idx==0), left (idx==1), and right (idx==2) image
        for img_idx in range(3):
            # get image name
            image_name = batch_sample[img_idx].split('/')[-1]
            
            # get image path
            img_path = os.path.join(self.path, 'IMG', image_name)
            
            # check if image exists
            if not os.path.isfile(img_path):
                print('Warning - Skipping image: No such file {img_path}'.format(img_path=img_path))
                continue
            
            # read in image and convert to rgb
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            images.append(image)

            # Data augmentation - Adjust steering angle
            if img_idx == 1:  # left image
                steering_angles.append(steering_angle + self.steering_offset)
            elif img_idx == 2:  # right image
                steering_angles.append(steering_angle - self.steering_offset)
            else:  # center image
                steering_angles.append(steering_angle) # no offset needed

            # Data augmentation - Flip center image and invert steering angle
            if img_idx == 0:  # center
                images.append(cv2.flip(image, 1))
                steering_angles.append(-steering_angle)

        return images, steering_angles
    
    def train(self, epochs=2, batch_size=32):
        """
        Runs training for the Behavorial Cloning model. The trained model is saved as 'model.h5'.
        """
        self.build_model()
        self.model.fit_generator(generator=self.generator(data=self.training_samples, batch_size=batch_size),
                                 validation_data=self.generator(data=self.validation_samples, batch_size=batch_size),
                                 epochs=epochs,
                                 steps_per_epoch=len(self.training_samples) * 10 // batch_size,
                                 validation_steps=len(self.validation_samples) // batch_size)
        
        self.model.save('model.h5')

def main():
    autodrive = AutonomousDriving()
    autodrive.import_data(path='./data', validation_size=0.15)
    autodrive.train(epochs=5, batch_size=128)  

if __name__ == '__main__':
    main()

