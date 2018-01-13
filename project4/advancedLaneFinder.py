import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from glob import glob
from proj4Tools import calibrate_camera, run_pipeline
from line import Line
from moviepy.editor import VideoFileClip
from proj4Tools import display_images
import cv2

# (1) Camera Calibration
mtx, dist = calibrate_camera()
n = 35
leftLine, rightLine = Line(n), Line(n)


def process_image(frame):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image where lines are drawn on lanes)

    result = run_pipeline(frame, mtx, dist, leftLine, rightLine)
    return result

def run_tests():
    # Remove the distortion from an image
    img = mpimg.imread('camera_cal/calibration1.jpg')
    undistImg = cv2.undistort(img, mtx, dist, None, mtx)

    # Test pipeline
    display_images(imgArr=[img, undistImg], imgLabels=['Original', 'Undistorted'], isImgGray=[0, 0], rows=1)
    images = glob('test_images/*.jpg')
    for fn in images:
        img = mpimg.imread(fn)
        plt.imshow(process_image(img))
        plt.show()

def run_test():
    # Test pipeline
    images = glob('test_images/*.jpg')
    fn = images[0]
    img = mpimg.imread(fn)
    result = process_image(img)
    plt.imshow(result)
    plt.show()

   # resultRGB = []
   # for row in result:
   #     newRow = [[val, val, val] for val in row]
   #     resultRGB.append(newRow)

   # return resultRGB

   # #print(resultRGB)

   # plt.imshow(resultRGB)
   # plt.show()

def run_frames(frames=[]):
    # Test pipeline
    #images = glob('frames/'+nameformat)
    for fn in frames:
        img = mpimg.imread(fn)
        outimg = process_image(img)

        print(img.shape)
        print(outimg.shape)
        display_images(imgArr=[img, outimg], imgLabels=['Original', 'Processed'], isImgGray=[0, 0], rows=1)


def run(input, output):

    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
    clip = VideoFileClip(input)#.subclip(38,45)#(10, 15) #(20,24)
    output_clip = clip.fl_image(process_image)

    output_clip.write_videofile(output, audio=False)


#run_test()
#run_tests()
#frames = ['frames/frame034.png']; run_frames(frames)
run(input='CarND-Advanced-Lane-Lines/project_video.mp4', output='./result_clip.mp4')

