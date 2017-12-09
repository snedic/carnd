# Import the data
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from generators import batchGenerator


samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if not reader.line_num == 1:
            samples.append(line)
train_samples, valid_samples = train_test_split(samples, test_size=0.2)
train_generator = batchGenerator(train_samples,batchSize=32, imgPath='./data/IMG/')
valid_generator = batchGenerator(valid_samples,batchSize=32, imgPath='./data/IMG/')


#images = []
#measurements = []
#for s in samples:
#    source_path = s[0]
#    filename = source_path.split('/')[-1]
#    current_path = './data/IMG/' + filename
#    image = cv2.imread(current_path)
#    images.append(image)
#    measurement = float(line[3])
#    measurements.append(measurement)
#
#X_train = np.array(images)
#y_train = np.array(measurements)



# Train a model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers import Cropping2D
from keras.layers import pooling, MaxPooling2D

#https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
model = Sequential()

#trim input images
model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)))

#normalization
model.add(Lambda(lambda x: (x / 127.5) - 1.))

#rest of the model
model.add(Convolution2D(6, 5, 5, activation="relu"))
#model.add(Convolution2D(36, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


#model.add(Convolution2D(24, 5, 5, activation="relu"))
#model.add(Convolution2D(36, 5, 5, activation="relu"))
##model.add(Convolution2D(48, 5, 5, activation="relu"))
##model.add(Convolution2D(64, 3, 3, activation="relu"))
##model.add(Convolution2D(64, 3, 3, activation="relu"))
#model.add(Flatten())
##model.add(Dense(100))
#model.add(Dense(50))
#model.add(Dense(10))
#model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                    validation_data=valid_generator, nb_val_samples=len(valid_samples),
                    nb_epoch=1)

model.save('model.h5', overwrite=True)
