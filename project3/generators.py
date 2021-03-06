from sklearn.utils import shuffle
from cv2 import imread, resize
from numpy import array, fliplr

def batchGenerator(samples, batchSize=33, resizeRatio=(1., 1.), imgPath='./data/IMG/'):
    ''' Generate a batch of size batchSize for the sample reference provided
        input:  samples - sample list containing a fileName at index 0 and center angle at index 3
                batchSize - the size of the desired batch, default is 32
                resizeRatio - resize the image by this ratio, default is (1.,1.) meaning no change
                imgPath - the directory path to the folder containing the images, default is ./data/IMG'''
    nSamples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, nSamples, batchSize):
            batchSamples = samples[offset:offset+batchSize]
            images = []
            angles = []

            for batchSample in batchSamples:
                name = imgPath+batchSample[0].split('/')[-1]
                centerImg = imread(name)
                centerAngle = float(batchSample[3])

                # Adjust steering measurements for the side camera images
                correction = 0.15#0.148 #0.2  # (angle/180), angle = ~26.6,

                leftName = imgPath+batchSample[1].split('/')[-1]
                leftImg = imread(leftName)
                leftAngle = centerAngle + correction

                rightName = imgPath+batchSample[2].split('/')[-1]
                rightImg = imread(rightName)
                rightAngle = centerAngle - correction

                # crop image
                #centerImg = centerImg[60:-22]

                # resize image
                #centerImg = resize(src=centerImg, dsize=(0, 0), fx=resizeRatio[0], fy=resizeRatio[1])

                images.append(centerImg)
                images.append(leftImg)
                images.append(rightImg)

                #images.append(fliplr(centerImg))
                #images.append(fliplr(leftImg))
                #images.append(fliplr(rightImg))

                angles.append(centerAngle)
                angles.append(leftAngle)
                angles.append(rightAngle)

                #angles.append(-centerAngle)
                #angles.append(-leftAngle)
                #angles.append(-rightAngle)

            # trim image to only see section with road
            X_train = array(images)
            y_train = array(angles)

            yield shuffle(X_train, y_train)
