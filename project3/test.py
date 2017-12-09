# Import the data
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from generators import batchGenerator
import numpy as np

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if not reader.line_num == 1:
            samples.append(line)
train_samples, valid_samples = train_test_split(samples, test_size=0.2)

img = batchGenerator(samples=train_samples, batchSize=2, resizeRatio=(0.6, 0.6))

a = (next(img))[0]


print(a.shape)