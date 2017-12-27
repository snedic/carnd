import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from glob import glob
from proj4Tools import calibrate_camera, run_pipeline
from line import Line
from moviepy.editor import VideoFileClip

# (1) Camera Calibration
mtx, dist = calibrate_camera()
leftLine, rightLine = Line(), Line()


def process_image(frame):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image where lines are drawn on lanes)

    result = run_pipeline(frame, mtx, dist, leftLine, rightLine)
    return result


def run_tests():
    images = glob('test_images/*.jpg')
    for fn in images:
        img = mpimg.imread(fn)
        plt.imshow(process_image(img))
        plt.show()


def run(input,output):

    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
    clip = VideoFileClip(input)
    output_clip = clip.fl_image(process_image)

    output_clip.write_videofile(output, audio=False)


#run_tests()
run(input='CarND-Advanced-Lane-Lines/project_video.mp4', output='./result.mp4')

