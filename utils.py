import os
import cv2

def add(a, b):
    return a + b

def convert_video_to_frames(source_path, output_path, step_size):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    vidcap = cv2.VideoCapture(source_path)
    success, image = vidcap.read()
    frame_index = 0
    success = True
    while success:
        # save frame as JPEG file
        cv2.imwrite(output_path + "/" + str(frame_index) + ".jpg", image)
        success, image = vidcap.read()
        print('Read a new frame: {0} {1}'.format(frame_index, success))
        frame_index += step_size