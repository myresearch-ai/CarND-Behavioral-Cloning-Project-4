# Behavioral Cloning for Self Driving Car

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This project is the fourth in the SDCND series - it builds upon computer vision fundamentals (image data manipulation & preprocessing) & **deep convolutional neural networks (D-CNN)** addressed in the previous project (project-3) where we predicted Germany traffic signs. The scope of this project is to develop a CNN that allows the car to achieve **perception** of its environment in order to autonomously navigate its environment (tracks). 

This project repo contains the following required files:

* model.py
* drive.py (modified)
* model.h5 & model.json
* video.mp4 (a video recording of the vehicle driving autonomously in its environment)

Goals
---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

Intallation & Resources
---
* Udacity [CarND-term-1 starter kit](https://github.com/udacity/CarND-Term1-Starter-Kit)with miniconda installation.
* Udacity [Car Simulation](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)
* Udacity [sample driving data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

Model Architeccture
---

The model's architecture is inpired by NVIDIA's architecture in the project [End to End Learning for Self-Driving Car](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 

![model](https://user-images.githubusercontent.com/76077647/129458962-9aeb7531-bacf-456d-ac64-a542d1d33b81.JPG)

#### Data Collection

The model used in this repository was trained on the [sample](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) data. The motivation behind the decision to only use udacity's sample data was concerned with data quality. It was rather challenging to drive **smoothely** around the tracks to generate high quality data for training a robust model. Given the size of the sample data, sufficient image manipulation & preprocessing was a must.

The data is generated from 3 cameras mounted on the vehicle which shows **left**, **center**, and **right** images as the vehicle navigates its environment. There are 4 features: **steering**, **throttle**, **brake**, and **speed** of which steering angle is the response variable.

![data](https://user-images.githubusercontent.com/76077647/129458691-f6ec376a-03fb-433f-8db5-8e3a997c083f.JPG)

#### Preprocessing

A preprocessing pipeline of multiple image data manipulation was implemented based on previous projects in this series & computer vision expertise. The utilities included cropping, warp perspective, resizing, normalization, augmentation, brightness adjustment, recovery, etc,. Recovery was particualrly hinted by udacity's lectures - the gist of it is concerned with steering the vehicle back to the center of the track when it goes off track. This was archieved using the left and right cameras as shown in the get_augmented_row() utility.
 
 ```
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
 ```
 
 The Model
 ---
 
 As inspired by NVIDIA's model, the model was implemented as below:
 
 ```
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
 ```
 
 The model was trained for 2 epochs on 80% of the data & tested on the remainder. The model's performance was exceptional scoring a 100% accuracy on the test set. 
 
 ![train_val](https://user-images.githubusercontent.com/76077647/129459050-7d60dec5-eb6a-4e7d-9a9a-aa4e344c4a59.JPG)
 
 #### Loading Model
 
 The command below is used to load the trained model for autonomous model of the simulator.
 
 ```
 python drive.py model.json
 ```
 
 Future Considerations
 ---
 
 * Generate more data using the simulator from both tracks for further model training
 * Experiment with more augmentation techniques

References & Credits
---
* [udacity self driving car nanodegree](https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7)
* [NVIDIA: End to End Learning for Self-Driving Car](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
