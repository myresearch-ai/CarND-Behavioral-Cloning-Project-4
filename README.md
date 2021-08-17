# Behavioral Cloning for Self Driving Car

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This project is the fourth in the SDCND series - it builds upon computer vision fundamentals (image data manipulation & preprocessing) & **deep convolutional neural networks (CNNs)** addressed in the previous project (project-3) where we predicted Germany traffic signs. The scope of this project is to develop a CNN that allows the car to achieve **perception** of its environment in order to autonomously navigate its environment (tracks). 

This project repo contains the following required files:

* model.py
* drive.py
* model.h5 & model.json
* video.mp4 (a video recording of the vehicle driving autonomously in its environment)
* requirements.txt (An option to the [CarND-term-1 starter kit](https://github.com/udacity/CarND-Term1-Starter-Kit) to set up environment to re-implement or run the pretrained model)

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

Basic image manipulation and preprocessing techniques were applied, following NVIDIA's approach, the following preprocessing techniques were applied;


|**Technique**|**Description/Comment**|
|-------------|---------------|
|*Normalization*| This was done using Keras Lambda layer|
|*Cropping*|Performed to remove less useful portions of video frames (images) that may negatively impact the model's performance|
|*RGB Transformation*|Images were converted to RGB format before feature extraction|
|*Augmentation*|Image flipping was performed for data augmentation|

Additionally, an offset factor was used to manipulate steering values to allow *smooth* driving of the car. Straight driving dominated the track, to prevent overfitting and biased training to straight driving, it was recommended to apply the offset.
 
 The Model
 ---
 
 As inspired by NVIDIA's model, the model was implemented as below:
 
 ```
  def build_model(self):
        """
        Model inspired by NVIDIA's architecture
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
 ```
 
 The model was trained for 5 epochs on 85% of the data & tested on the remainder. The model's performance on the validation set was impressive. 

 
 #### Loading Model
 
 The command below is used to load the trained model for autonomous model of the simulator.
 
 ```
 python drive.py model.h5
 ```
 
 Future Considerations
 ---
 
 * Generate more data using the simulator from both tracks for further model training
 * Experiment with more augmentation techniques
 * Apply additional preprocessing techniques such as bright contrast, perspective transform, etc.

References & Credits
---
* [udacity self driving car nanodegree](https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7)
* [NVIDIA: End to End Learning for Self-Driving Car](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
