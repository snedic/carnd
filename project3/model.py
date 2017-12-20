# Import the data
import csv
from sklearn.model_selection import train_test_split
from generators import batchGenerator

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers import pooling, MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Dropout

from helpers import plotLossHistory

samples = []
with open('./trainData/driving_log_2017.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if not reader.line_num == 1:
            samples.append(line)
train_samples, valid_samples = train_test_split(samples, test_size=0.2)
#resizeImgRatio = (0.5, 0.5)
train_generator = batchGenerator(train_samples,
                                 batchSize=33,
                                 #resizeRatio=resizeImgRatio,
                                 imgPath='./trainData/IMG/')
valid_generator = batchGenerator(valid_samples,
                                 batchSize=33,
                                 #resizeRatio=resizeImgRatio,
                                 imgPath='./trainData/IMG/')

# Train a model
row, col, ch = 160, 320, 3#39, 160, 3#58, 240, 3
dropRate = 0.0
#https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
model = Sequential()

#crop
model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(row, col, ch)))

#normalization
#model.add(Lambda(lambda x: (x / 127.5) - 1.))#, input_shape=(row, col, ch)))
model.add(Lambda(lambda x: (x / 255.0) -0.5))#, input_shape=(160,320,3)))
#
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))

#rest of the model
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(dropRate))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(dropRate))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(dropRate))
model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation="relu"))
model.add(Dropout(dropRate))
model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation="relu"))
model.add(Dropout(dropRate))

#model.add(MaxPooling2D())
#`model.add(Dropout(dropRate))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(dropRate))
model.add(Dense(50))
model.add(Dropout(dropRate))
model.add(Dense(10))
model.add(Dropout(dropRate))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*3*2,
                    validation_data=valid_generator, nb_val_samples=len(valid_samples)*3*2,
                    nb_epoch=5)

model.save('nvidia.h5', overwrite=True)

plotLossHistory(history_object, saveFileName='msePerEpoch_myModel.png')
