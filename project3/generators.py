from sklearn.utils import shuffle
from cv2 import imread, resize
from numpy import array

def batchGenerator(samples, batchSize=32, resizeRatio=(1., 1.), imgPath='./data/IMG/'):
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

                # crop image
                centerImg = centerImg[60:-22]

                # resize image
                centerImg = resize(src=centerImg, dsize=(0, 0), fx=resizeRatio[0], fy=resizeRatio[1])

                images.append(centerImg)
                angles.append(centerAngle)


            # trim image to only see section with road
            X_train = array(images)
            y_train = array(angles)

            yield shuffle(X_train, y_train)