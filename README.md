# Use-Deep-Learning-to-Clone-Driving-Behavior
Built and trained a convolutional neural network for end-to-end driving in a simulator, using TensorFlow and Keras. Used optimization techniques such as regularization and dropout to generalize the network for driving on multiple tracks.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Results
#### Autonomous Driving on Basic Track
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/OB41L8yqzGs/0.jpg)](https://www.youtube.com/watch?v=OB41L8yqzGs)
#### Autonomous Driving on Advanced Track
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/PwI7Fr9t5qQ/0.jpg)](https://www.youtube.com/watch?v=PwI7Fr9t5qQ)


[//]: # (Image References)

[image1]: ./examples/1.png "1"
[image2]: ./examples/2.png "2"
[image3]: ./examples/3.png "3"

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md  summarizing the results

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network used by NVIDIA (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) (model.py lines 14-49)![alt text][image1]

The model includes RELU layers to introduce nonlinearity (code line 19), and the data is normalized in the model using a Keras lambda layer (code line 17). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting (model.py lines 46). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 56-59). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 53).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, two sets of complete laps of both forward and opposite driving for both tracks

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow Udacity's recommended model, which is the Nvidia's architecture

My first step was to use a convolution neural network model similar to the Nvidia's. From this model, I added lambda layer to normalize images and dropout layer to avoid overfitting. Maxpooling layer with size 2 by 2 was also added after every convolutional layer.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. After training the model, a graph was plotted to observe model loss. The training loss was higher than validation loss for all epochs, which indicated the model was not overfitting.

Since the track data was limited, I decided augmenting data would be a good idea, just like the Traffic Sign Classification project. The images were taken in randomly and they are jittered. The images were shifted horizontally to augment input data, cropped and resized to 64 by 64 as model input. 

The final step was to run the simulator to see how well the car was driving around track one. Everything seemed good except the car was 'wavy' at some points of the track. The car was provided with a constant throttle of 0.3 in drive.py. As for the second track, drive.py had to be modified to mach throttle with the road condition. I used a couple of if statements to control the car's speed to I will stay on track. The car was able to complete track 1 flawlessly. On the second track, the car got through most of the track except the last curvy slope right next to the starting point.

#### 2. Final Model Architecture

The final model architecture is summarized below:
![alt text][image2]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Then I recorded one lap driving backwards. Lastly, the car was placed on the side of the road, pressed record, and drove the car back to the center of the road as a recovery lap. This same process was repeated for track two.

To augment the data sat, the images were jittered horizontally to create 'curvy road' effect. 

After the collection process, I had 18035 data points. I then preprocessed this data by cropping and resizing. 

I then randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8. ![alt text][image3] Starting from 9th epoch, the validation loss began to climb to the point where the model looked like it was overfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
