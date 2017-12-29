import cv2


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    for bbox in bboxes:
        cv2.rectangle(draw_img, bbox[1], bbox[0], color, thick)
    # return the image copy with boxes drawn
    return draw_img # Change this line to return image copy with boxes