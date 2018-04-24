import os
import sys

sys.path.append(os.path.dirname(__file__) + "/pose-tensorflow/")

from scipy.misc import imread

from config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input

def detect_pose(image_paths):
    cfg = load_config("./pose_cfg.yaml")

    # Load and setup CNN part detector
    sess, inputs, outputs = predict.setup_pose_prediction(cfg)

    poses = []
    for image_path in image_paths:
        # Read image from file
        image = imread(image_path, mode='RGB')

        image_batch = data_to_input(image)

        # Compute prediction with the CNN
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

        # Extract maximum scoring location from the heatmap, assume 1 person
        pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
        poses.append(pose)

    return poses, cfg