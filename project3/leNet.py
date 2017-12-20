# Import the data
import csv
from sklearn.model_selection import train_test_split
from generators import batchGenerator

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Dropout

from helpers import plotLossHistory

samples = []
with open('./trainData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if not reader.line_num == 1:
            samples.append(line)
train_samples, valid_samples = train_test_split(samples, test_size=0.2)
train_generator = batchGenerator(train_samples,
                                 batchSize=33,
                                 imgPath='./trainData/IMG/')
valid_generator = batchGenerator(valid_samples,
                                 batchSize=33,
                                 imgPath='./trainData/IMG/')

# Train a model
row, col, ch = 160, 320, 3#39, 160, 3#58, 240, 3
dropRate = 0.25

model = Sequential()

#crop
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(row, col, ch)))

#normalization
model.add(Lambda(lambda x: (x / 255.0) - 0.5))#127.5) - 1.))

#leNet
model.add(Convolution2D(6, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(dropRate))
model.add(MaxPooling2D())
model.add(Dropout(dropRate))
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(dropRate))
model.add(Dense(84))
model.add(Dropout(dropRate))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*3*2,
                    validation_data=valid_generator, nb_val_samples=len(valid_samples)*3*2,
                    nb_epoch=10)

model.save('leNet.h5', overwrite=True)

plotLossHistory(history_object, saveFileName='msePerEpoch_leNet.png')
