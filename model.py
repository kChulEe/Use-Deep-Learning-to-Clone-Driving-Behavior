import numpy as np 
import pandas as pd
import cv2
import random
import os
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout

#network architecture used by NVIDIA as recommended
#https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
def model():
    model = Sequential()
    #lambda layer to normalize
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(64,64,3), output_shape=(64,64,3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    #added dropout layer to avoid overfitting of the model
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    return model

model = model()

model.compile(loss='mse', optimizer=Adam(lr=1e-4))

#to split the original data into training and validation
def split_data(csv_file, proportion):
    shuffled_data = csv_file.iloc[np.random.permutation(len(csv_file))]
    validation_data = int(len(csv_file) * proportion)
    return(shuffled_data[validation_data:], shuffled_data[:validation_data])

#read csv file using pandas and split trainig from validation data
csv_file = pd.read_csv('./training_data/driving_log.csv')
training_data, validation_data = split_data(csv_file, 0.2)

#to crop out the unnecessary parts of the input image
def crop(image):
    image_shape = image.shape 
    output_image = image[60:135, 0: image_shape[1], :]
    return output_image

#resize images into smaller sizes for the network input
def resize(image, resize_shape):
    return cv2.resize(image, resize_shape, cv2.INTER_AREA)

column_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

#input images randomly, correct image paths and camera correction for steering measurements (taken from Udacity)
def random_data(data, row):
    random = np.random.randint(0,3)
    image = data.iloc[row][column_names.index('center') + random].strip()
    image_path = image.split('/')[-1]
    image = './training_data/IMG/' + image_path    
    measurement = data.iloc[row][column_names.index('steering')]

    if random == column_names.index('left'):
        measurement = measurement + 0.2
    elif random == column_names.index('right'):
        measurement = measurement - 0.2
        
    return(image, measurement)

#uses above method to randomize input images
def preprocess_data(data, data_size=64):
    random_rows = np.random.randint(0, len(data), data_size)
    x = []
    for row in random_rows:
        output = random_data(data, row)
        x.append(output)
    return x

#top portion of the image is moved horizontally to augment input data
#credit to https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
def process_jittering(image, measurement, range=150):
    height, width, depth = image.shape
    x = np.random.randint(-range, range + 1)
    y = [width / 2 + x, height / 2]
    y1 = np.float32([[0, height], [width, height], [width / 2, height / 2]])
    y2 = np.float32([[0, height], [width, height], y])
    new_measurement = x / (height / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    z = cv2.getAffineTransform(y1, y2)
    image = cv2.warpAffine(image, z, (width, height), borderMode=1)
    measurement += new_measurement
    return image, measurement

#process jittering
def jitter_images(image, measurement, resize_shape, prob=0.5):
    if random.random() < prob:
        image, measurement = process_jittering(image, measurement)
    return image, measurement

#use generator to feed in model after jittering
def process_data(data, augment=True, resize_shape=(64,64), data_size=64):
    while True:
        X_data = []
        y_data = []
        images = preprocess_data(data, data_size)
        for image_file, measurement in images:
            img = plt.imread(image_file)
            if augment:
                img, measurement = jitter_images(img, measurement, resize_shape)
          
            img = crop(img)
            img = resize(img, resize_shape)
            X_data.append(img)
            y_data.append(measurement)
        yield np.array(X_data), np.array(y_data)

generated_training_data = process_data(training_data, resize_shape=(64,64))
print('Training Data Generated')
generated_validation_data = process_data(validation_data, resize_shape=(64,64))
print('Validation Data Generated')

print(len(training_data))
print(len(validation_data))

history = model.fit_generator(generated_training_data, 
                              samples_per_epoch=9984,
                              nb_epoch=8, 
                              validation_data=generated_validation_data, 
                              nb_val_samples=4508,
                              verbose=1)

#provided by udacity
model_json = model.to_json()
with open('model.json', "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')

#display loss graph using history instance
print(history.history.keys())

import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


