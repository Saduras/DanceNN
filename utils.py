import os
import sys
import time
import cv2

import numpy as np
import pandas as pd
import moviepy.editor as mpe
sys.path.append(os.path.dirname(__file__) + "/pose-tensorflow/")

from config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input

def predict_frame(video, t):
    frame_count = int(video.fps * video.duration)
    frame_length = 1 / video.fps

    # Load and setup CNN part detector
    cfg = load_config('./pose_cfg.yaml')
    sess, inputs, outputs = predict.setup_pose_prediction(cfg)

    frame = video.get_frame(t)

    image_batch = data_to_input(frame)

    # Compute prediction with the CNN
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

    # Extract maximum scoring location from the heatmap, assume 1 person
    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)

    return pose

def preprocess(video_name, duration):
    source_path = f'./data/video/{video_name}.mp4'

    csv_base_path = './data/poses/'
    if not os.path.exists(csv_base_path):
        os.makedirs(csv_base_path)
    csv_path = f'{csv_base_path}{video_name}_poses.csv'

    audio_base_path = './data/audio/'
    if not os.path.exists(audio_base_path):
        os.makedirs(audio_base_path)
    audio_path = f'{audio_base_path}{video_name}.mp3'

    start_time = time.time()

    video = mpe.VideoFileClip(source_path)
    if duration < 0:
        duration = video.duration

    frame_count = int(video.fps * duration)
    frame_length = 1 / video.fps
    print(f'video length: {video.duration}s fps: {video.fps} frame count: {frame_count}')

    # Load and setup CNN part detector
    cfg = load_config('./pose_cfg.yaml')
    sess, inputs, outputs = predict.setup_pose_prediction(cfg)
    print('pose model loaded')

    poses = []
    times = []
    for i in range(frame_count):
        t = i * frame_length
        frame = video.get_frame(t)

        image_batch = data_to_input(frame)

        # Compute prediction with the CNN
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

        # Extract maximum scoring location from the heatmap, assume 1 person
        pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
        poses.append(pose)
        times.append(t)

        if i % 100 == 0:
            print(f'processed frame: {i}/{frame_count} elapsed time: {time.time() - start_time}')

    sess.close()
    print(f'saving poses at {csv_path}')
    save_poses(np.array(poses), times, cfg, csv_path)
    print(f'saving audio at {audio_path}')
    video.audio.write_audiofile(audio_path)
    print(f'total time: {time.time() - start_time}')

def save_poses(poses, times, cfg, output_path):
    dict = {}
    dict['time'] = times
    cols = ['time']
    for j in range(len(cfg.all_joints)):
        prefix = cfg.all_joints_names[j]
        for i in range(len(cfg.all_joints[j])):
            idx = cfg.all_joints[j][i]
            
            label_x = prefix + '_' + str(i) + '_x'
            dict[label_x] = poses[:,idx,0]
            cols.append(label_x)

            label_y = prefix + '_' + str(i) + '_y'
            dict[label_y] = poses[:,idx,1]
            cols.append(label_y)

    df = pd.DataFrame(data = dict, columns = cols)
    df.to_csv(output_path, index=False)


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