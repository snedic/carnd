import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, n=20):
        self.n = n
        self.skippedFrames = 0

        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30. / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_xfitted = []

        #average x values of the fitted line over the last n iterations
        self.bestx = None

        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        self.previous_fit = [np.array([False])]

        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        self.previous_radius_of_curvature = None

        #distance in meters of vehicle center from the line
        self.line_base_pos = None

        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        self.previous_diffs = self.diffs
        self.recent_diffs = []

        #x values for detected line pixels
        self.previous_xy = None
        self.allx = None

        #y values for detected line pixels
        self.ally = None

    def appendXFitted(self, x_fit):
        self.recent_xfitted.append(x_fit)
        if len(self.recent_xfitted) > self.n:
            self.recent_xfitted.pop(0)

    def append_prev_diffs(self, x_fit):
        self.recent_diffs.append(x_fit)
        if len(self.recent_diffs) > self.n:
            self.recent_diffs.pop(0)

    #def appendFittedCoeffs(self, curr_fit):
    #    self.recent_fitted.append(curr_fit)
    #    if len(self.recent_fitted) > self.n:
    #        self.recent_fitted.pop(0)

    def get_curve_radius(ploty, curr_fit):
        #ploty = np.linspace(0, warpedImage.shape[0] - 1, warpedImage.shape[0])
        y_eval = np.max(ploty)

        curverad = ((1 + (2 * curr_fit[0] * y_eval + curr_fit[1]) ** 2) ** 1.5) / np.absolute(2 * curr_fit[0])

        return curverad

    def get_scaled_curve_radius(self, ploty, fitx):
        y_eval = np.max(ploty)

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(ploty * self.ym_per_pix, fitx * self.xm_per_pix, 2)

        # Calculate the new radii of curvature
        curverad = ((1 + (2 * fit_cr[0] * y_eval * self.ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

        return curverad


    def frame_skipped(self):
        #print("Frame Skipped")
        self.skippedFrames += 1

        # remove last added frame from the arrays
        self.recent_xfitted.pop()
        self.recent_diffs.pop()

        # check if we need to reassess the lane detection with a sliding window
        self.detected = self.skippedFrames > 5# self.n*.5 #self.skippedFrames < self.n/2

        # reset current values
        self.current_fit = self.previous_fit
        self.diffs = self.previous_diffs
        self.allx, self.ally = self.previous_xy
        self.bestx = np.mean(self.allx)
        self.best_fit = np.mean(self.recent_xfitted, axis=0)
        self.radius_of_curvature = self.previous_radius_of_curvature

    def store_values(self, binary_img, curr_fit, curr_xy):
        ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
        fitx = curr_fit[0] * ploty ** 2 + curr_fit[1] * ploty + curr_fit[2]

        diffs = np.subtract(curr_fit, self.current_fit)
        self.detected = True

        # store current & previous fit
        self.previous_fit = self.current_fit
        self.current_fit = curr_fit

        # update differences
        self.previous_diffs = self.diffs
        self.diffs = diffs
        self.append_prev_diffs(self.diffs)

        # store detected line pixels
        self.previous_xy = [self.allx, self.ally]
        self.allx, self.ally = curr_xy

        # store the fit lines
        self.appendXFitted(fitx)#curr_fit)

        # update the average best fit value
        self.bestx = np.mean(self.allx)

        # update the average best fit coefficients
        self.best_fit = np.mean(self.recent_xfitted, axis=0)

        # set the radius of curvature
        self.previous_radius_of_curvature = self.radius_of_curvature
        self.radius_of_curvature = self.get_scaled_curve_radius(ploty, fitx)

        # store distance (m) of vehicle center to the line
        base_pos = binary_img.shape[1] / 2
        self.line_base_pos = (base_pos - fitx[-1]) * self.xm_per_pix

